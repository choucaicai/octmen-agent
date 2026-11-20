"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions, inheriting
from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained, but exactly replicate the
logic in `prismatic.models.vlms.prismatic.py`.

Note =>> for the time being, not adding the custom HF "docstring" formatting.

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
"""

import logging
from .trajectory_memory import AdaptiveOctreeMemory, MemoryState, block_default_keys
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig
from .modeling_rgbdtower import RGBDSceneTower, cartesian_to_spherical
from .modeling_explore import ExploraTokenModel
# Get Logger
logger = logging.getLogger(__name__)

# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


def fuse_voxel_memory(state, keys, current, xyz_sum, counts, step_id):
    """Fuse pooled observations with the previous snapshot before saving."""
    if state is None or not len(state.keys):
        return MemoryState(keys, current, xyz_sum, counts, step_id)
    if not len(keys):
        return replace(state, step_id=step_id)

    candidates, inverse = torch.unique(
        torch.cat((state.keys, keys, block_default_keys(keys))), dim=0, return_inverse=True,
    )
    old_rows, current_rows, parent_rows = inverse.split((len(state.keys), len(current), len(current)))
    old_indices = old_rows.new_full((len(candidates),), -1)
    old_indices[old_rows] = torch.arange(len(old_rows), device=old_rows.device)
    exact, parent = old_indices[current_rows], old_indices[parent_rows]
    previous = torch.where(exact >= 0, exact, parent)
    averages = (state.features[previous.clamp_min(0)].float() + current.float()) * 0.5
    updated = torch.where(previous[:, None] >= 0, averages.to(current.dtype), current)

    stored = (old_indices >= 0).index_fill(0, current_rows, True)
    compact = stored.cumsum(0, dtype=torch.long) - 1
    keys = candidates[stored]
    old_rows, current_rows = compact[old_rows], compact[current_rows]
    hidden = current.new_zeros((len(keys), current.shape[-1])).index_copy(0, old_rows, state.features)
    hidden = hidden.index_copy(0, current_rows, updated)
    xyz_sum = xyz_sum.new_zeros((len(keys), 3)).index_copy(0, old_rows, state.xyz_sum).index_add(0, current_rows, xyz_sum)
    counts = counts.new_zeros(len(keys)).index_copy(0, old_rows, state.counts).index_add(0, current_rows, counts)
    return MemoryState(keys, hidden, xyz_sum, counts, step_id)


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper

# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
        return torch.cat([patches, patches_fused], dim=2)

# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa




class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    memory_config_keys = ('memory_voxel_size',)
    auxiliary_module_names = ('explore_model', 'oct_model')

    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        config.memory_voxel_size = float(getattr(config, 'memory_voxel_size', 5.0))
        self.trajectory_memory = AdaptiveOctreeMemory()
        self.voxel_size = config.memory_voxel_size
        
        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )
        
        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size

        self.pad_token_id = config.pad_token_id

        self.num_query_tokens = int(getattr(config, 'num_query_tokens', 128))
        self.num_frontier_tokens = int(getattr(config, 'num_frontier_tokens', 16))
        
        self.oct_model = RGBDSceneTower(latent_dim=self.vision_backbone.embed_dim,voxel_size=self.voxel_size)
        self.qformer_hidden_size = self.vision_backbone.embed_dim
        self.explore_model = ExploraTokenModel(num_query_tokens=self.num_query_tokens,num_frontier_tokens=self.num_frontier_tokens,qformer_hidden_size=self.qformer_hidden_size, vision_hidden_size=self.vision_backbone.embed_dim, llm_dim=config.text_config.hidden_size)

        self.clip_tokens = 1024
        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    def initialize_explore_model(self, device='cuda'):
        print('creating explore_model')
        self.explore_model = ExploraTokenModel(
            num_query_tokens=self.num_query_tokens,
            num_frontier_tokens=self.num_frontier_tokens,
            qformer_hidden_size=self.qformer_hidden_size,
            vision_hidden_size=self.vision_backbone.embed_dim,
            llm_dim=self.config.text_config.hidden_size,
        ).to(device)
        self.explore_model.init_weights()

    def initialize_oct_model(self, device='cuda'):
        print('creating oct_model')
        self.oct_model = RGBDSceneTower(latent_dim=self.vision_backbone.embed_dim,voxel_size=self.voxel_size).to(device)

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()
    
    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    @property
    def _memory_states(self):
        return self.trajectory_memory.states

    @_memory_states.setter
    def _memory_states(self, states):
        self.trajectory_memory.states = states

    def reset_memory(self, episode_id=None):
        self.trajectory_memory.reset(episode_id)

    def detach_memory(self):
        self.trajectory_memory.detach()

    def prepare_explore_rgbd_memory(self, image_features, depths, params, poses, episode_id=None, step_id=None):
        """Update one trajectory and return its memory for a later batched read."""
        if episode_id is None or step_id is None:
            raise ValueError("model_octmem_agent requires episode_ids and consecutive step_ids")
        step_id = int(step_id)
        state = self.trajectory_memory.get(episode_id)
        if state is None or state.step_id != step_id:
            last_step = -1 if state is None else state.step_id
            if step_id != last_step + 1:
                raise ValueError(f'Nonconsecutive memory step: previous={last_step}, current={step_id}')
            batch_size, num_view = image_features.shape[:2]
            image_features = image_features.flatten(0, 1)
            patches = image_features.shape[1]
            side = int(patches ** 0.5)
            if side * side != patches:
                raise ValueError("Visual tokens must form a square patch grid")
            features = image_features.permute(0, 2, 1).reshape(batch_size, num_view, -1, side, side)
            visual, xyz, valid, fine = self.oct_model.unproject_current(features, depths, params, poses)
            mask = valid[0]
            pooled = self.oct_model.pool_current_voxels(visual[0, mask], xyz[0, mask], fine[0, mask])
            state = fuse_voxel_memory(state, *pooled, step_id=step_id)
            self.trajectory_memory.save(episode_id, state)
        tokens, world_xyz, coarse = self.trajectory_memory.read(state)
        geometry = cartesian_to_spherical(world_xyz.reshape(1, 1, 1, -1, 3), poses[0, -1].float()).reshape(-1, 6)
        position_features = torch.cat((tokens, geometry.to(tokens.dtype)), dim=-1)
        return self.oct_model.encode_features_with_xyz(position_features), coarse & (geometry[:, 3] > 40)

    def forward_memory_window(self, steps):
        """One distributed forward/backward graph for up to two consecutive steps."""
        if not 1 <= len(steps) <= 2:
            raise ValueError("This entry point implements TBPTT windows of at most two steps")
        weighted_losses, counts = [], []
        for batch in steps:
            output = self.forward(**batch, return_dict=True, use_cache=False)
            count = batch["labels"][:, 1:].ne(IGNORE_INDEX).sum()
            if count.item() == 0:
                raise ValueError("Window step has no supervised action tokens")
            weighted_losses.append(output.loss.float() * count)
            counts.append(count)
            for key, is_last in zip(batch["episode_ids"], batch["episode_ends"]):
                if bool(is_last):
                    self.reset_memory(key)
        token_count = torch.stack(counts).sum()
        return {"loss": torch.stack(weighted_losses).sum() / token_count,
                "token_count": token_count}

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,

        camera_depths: Optional[torch.FloatTensor] = None,
        camera_poses: Optional[torch.FloatTensor] = None,
        camera_params: Optional[torch.FloatTensor] = None,
        camera_image_features: Optional[torch.FloatTensor] = None,

        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        memory_window=None,
        episode_ids=None,
        step_ids=None,
        episode_ends=None,
        
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        if memory_window is not None:
            return self.forward_memory_window(memory_window)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training
        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None
        
        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # === Handle Multimodal Forward ===
        elif (
            pixel_values is not None
            and (
                (input_ids is not None and input_ids.shape[0] == pixel_values.shape[0])
                or (inputs_embeds is not None and inputs_embeds.shape[0] == pixel_values.shape[0])
            )
        ):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"
            # Visual Feature Extraction
            rgbd_features = []
            input_embeddings = self.get_input_embeddings()(input_ids)
            prompt_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
            if labels is not None:
                prompt_mask = prompt_mask & labels.eq(IGNORE_INDEX)
            prompt_embeddings = input_embeddings.masked_fill(~prompt_mask.unsqueeze(-1), 0)
            prompt_mean = prompt_embeddings.sum(dim=1, dtype=torch.float32)
            prompt_mean = (prompt_mean / prompt_mask.sum(dim=1, keepdim=True).clamp_min(1)).to(input_embeddings.dtype)
            
            # DeepSpeed BF16 does not supply the native autocast used by plain DDP.
            pixel_values = pixel_values.to(dtype=next(self.vision_backbone.parameters()).dtype)
            current_visual = self.vision_backbone(pixel_values)
            cur_rgb_features = self.projector(current_visual)
            batch_size = pixel_values.shape[0]
            if episode_ids is None or step_ids is None:
                raise ValueError("model_octmem_agent requires episode_ids and consecutive step_ids")
            if len(episode_ids) != batch_size or len(step_ids) != batch_size:
                raise ValueError("Episode metadata must match the image batch size")
            if len(set(str(key) for key in episode_ids)) != batch_size:
                raise ValueError("A batch must contain distinct episodes; recurrent time steps run sequentially")
            if any(len(items) != batch_size for items in (camera_depths, camera_params, camera_poses)):
                raise ValueError("Camera geometry must match the image batch size")
            if camera_image_features is not None and len(camera_image_features) != batch_size:
                raise ValueError("Precomputed image features must match the image batch size")

            memory_entries = []
            for b_i, (cur_depths, cur_params, cur_poses) in enumerate(zip(camera_depths, camera_params, camera_poses)):
                cur_image_features = (current_visual[b_i:b_i + 1] if camera_image_features is None
                                      else camera_image_features[b_i])
                if any(len(value) != 1 for value in (cur_image_features, cur_depths, cur_params, cur_poses)):
                    raise ValueError("Each episode must supply exactly one new frame per environment step")
                memory_entries.append(self.prepare_explore_rgbd_memory(
                    cur_image_features.unsqueeze(0), cur_depths.unsqueeze(0), cur_params.unsqueeze(0),
                    cur_poses.unsqueeze(0), episode_id=episode_ids[b_i], step_id=int(step_ids[b_i]),
                ))

            # One Q-Former call per time step, with padding hidden from both query groups.
            batched_rgbd_features = self.explore_model.forward_memory_batch(
                memory_entries, prompt_mean,
            )
            for b_i, rgbd_feature in enumerate(batched_rgbd_features):
                if rgbd_feature.shape[0] > self.clip_tokens:
                    indices = torch.randperm(rgbd_feature.size(0), device=rgbd_feature.device)[:self.clip_tokens]
                    rgbd_feature = rgbd_feature[indices]
                rgbd_features.append(torch.cat([cur_rgb_features[b_i], rgbd_feature]))

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            
            if labels is None:
                labels = torch.full_like(input_ids, IGNORE_INDEX)

            # Insert memory tokens after BOS, then pad the variable-length sequences.
            new_input_embeds = []
            new_labels = []
            batch_size = input_ids.shape[0]
            for batch_idx, cur_input_ids in enumerate(input_ids):
                cur_rgbd_features = rgbd_features[batch_idx]
                cur_embeddings = torch.cat(
                    [input_embeddings[batch_idx, :1, :], cur_rgbd_features, input_embeddings[batch_idx, 1:, :]]
                )      
                cur_rgbd_labels = torch.full(
                    (cur_rgbd_features.shape[0],),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                cur_labels = torch.cat([labels[batch_idx, :1], cur_rgbd_labels, labels[batch_idx, 1:]],)
                
                new_input_embeds.append(cur_embeddings)
                new_labels.append(cur_labels)

            max_len = max(x.shape[0] for x in new_input_embeds)
            new_input_embeds_padded = []
            new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
            attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)

            for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
                cur_len = cur_new_embed.shape[0]
                if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                    new_input_embeds_padded.append(torch.cat((
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                        cur_new_embed
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, -cur_len:] = cur_new_labels
                        attention_mask[i, -cur_len:] = True
                else:
                    new_input_embeds_padded.append(torch.cat((
                        cur_new_embed,
                        torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                    ), dim=0))
                    if cur_len > 0:
                        new_labels_padded[i, :cur_len] = cur_new_labels
                        attention_mask[i, :cur_len] = True
            
            new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
            new_labels = new_labels_padded
            # Dispatch to Language Model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=new_input_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif pixel_values is not None and input_ids.shape[0] != pixel_values.shape[0]:
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )
        
    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        camera_depths: Optional[torch.FloatTensor] = None,
        camera_poses: Optional[torch.FloatTensor] = None,
        camera_params: Optional[torch.FloatTensor] = None,
        camera_image_features: Optional[torch.FloatTensor] = None,

        episode_ids=None,
        step_ids=None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")
        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "camera_depths": camera_depths ,
                "camera_poses": camera_poses ,
                "camera_params": camera_params ,
                "camera_image_features": camera_image_features,
                "episode_ids": episode_ids,
                "step_ids": step_ids,
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of
        
        self.post_init()

    def predict_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        # Run VLA inference
        generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
