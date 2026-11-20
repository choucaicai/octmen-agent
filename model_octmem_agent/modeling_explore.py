import torch
import torch.nn as nn
from .multimodal_blip2.configuration_blip_2 import Blip2QFormerConfig
from .multimodal_blip2.modeling_blip_2 import Blip2QFormerModel
from typing import Optional
from torch.nn.utils.rnn import pad_sequence


class ExploraTokenModel(nn.Module):
    def __init__(self,num_query_tokens=64,num_frontier_tokens=32,qformer_hidden_size=768,vision_hidden_size=768,num_attention_heads=16,llm_dim=4096):
        super().__init__()
    
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, qformer_hidden_size))
        self.frontier_tokens =  nn.Parameter(torch.zeros(1, num_frontier_tokens, qformer_hidden_size))
        self.null_memory = nn.Parameter(torch.zeros(1, 1, vision_hidden_size))
        
        qformer_config = Blip2QFormerConfig(
            hidden_size=qformer_hidden_size,
            num_attention_heads=num_attention_heads,
            encoder_hidden_size = vision_hidden_size,
            num_hidden_layers = 16,
        )
        self.scale = nn.Linear(llm_dim, qformer_hidden_size)
        self.shift = nn.Linear(llm_dim, qformer_hidden_size)
        
        self.qformer = Blip2QFormerModel(qformer_config)
        self.qformer_projector = nn.Sequential(
            nn.Linear(qformer_hidden_size, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        total_params = sum(p.numel() for p in self.qformer.parameters())
        print(f"#### former total params: {total_params / 1e6:.2f}M",self.query_tokens.shape,self.frontier_tokens.shape)
        self.init_weights()
        
    def init_weights(self):
        def _init_weights(module):
            """Initialize the weights"""
            factor = 0.02
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=factor)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        
        self.qformer.apply(_init_weights)
        
        self.qformer_projector.apply(_init_weights)
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.frontier_tokens, std=0.02)
        nn.init.normal_(self.null_memory, std=0.02)

    def forward_memory_batch(self, memory_entries, average_language_embeddings):
        tokens, far_masks = zip(*memory_entries)
        memory = pad_sequence(tokens, batch_first=True)
        far_mask = pad_sequence(far_masks, batch_first=True, padding_value=False)
        other_mask = pad_sequence(tuple(~mask for mask in far_masks), batch_first=True, padding_value=False)
        return self.forward_exp(memory, average_language_embeddings, other_mask, far_mask)
        
    def forward_exp(
                self,
                memory_feats,
                average_language_embedding,
                near_mask,
                far_mask,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
        ):
        batch_size = memory_feats.shape[0]
        memory_shape = memory_feats.shape[:2]

        num_scene_queries = self.query_tokens.shape[1]
        num_explore_queries = self.frontier_tokens.shape[1]

        gamma = self.scale(average_language_embedding)
        beta = self.shift(average_language_embedding)

        scene_queries = self.query_tokens.expand(batch_size, -1, -1)
        scene_queries = scene_queries * (1 + gamma[:, None, :]) + beta[:, None, :]
        explore_queries = self.frontier_tokens.expand(batch_size, -1, -1)
        explore_queries = explore_queries * (1 + gamma[:, None, :]) + beta[:, None, :]
        query_tokens = torch.cat([scene_queries, explore_queries], dim=1)

        
        near_mask = near_mask.to(device=memory_feats.device, dtype=torch.bool)
        far_mask = far_mask.to(device=memory_feats.device, dtype=torch.bool)
        if near_mask.dim() != 2 or far_mask.dim() != 2:
            raise ValueError("near_mask and far_mask must have shape [B, Nv].")
        if near_mask.shape != memory_shape or far_mask.shape != memory_shape:
            raise ValueError("near_mask/far_mask shape must match memory_feats[:, :, 0].")

        null_memory = self.null_memory.expand(batch_size, -1, -1).to(dtype=memory_feats.dtype)
        memory_feats = torch.cat([memory_feats, null_memory], dim=1)
        null_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=memory_feats.device)
        near_mask = torch.cat([near_mask, null_valid], dim=1)
        far_mask = torch.cat([far_mask, null_valid], dim=1)

        scene_cross_mask = (near_mask | far_mask)[:, None, :].expand(-1, num_scene_queries, -1)
        explore_cross_mask = far_mask[:, None, :].expand(-1, num_explore_queries, -1)
        cross_mask = torch.cat([scene_cross_mask, explore_cross_mask], dim=1).to(dtype=memory_feats.dtype)

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=memory_feats,
            encoder_attention_mask=cross_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        return self.qformer_projector(query_outputs[0])
