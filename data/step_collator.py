"""Pad action token sequences and batch current-frame observations."""
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100
CAMERA_KEYS = ("camera_depths", "camera_params", "camera_poses")


@dataclass
class PaddedCollatorForActionPrediction:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __call__(self, instances):
        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence([sample["input_ids"] for sample in instances], batch_first=True,
                                 padding_value=self.pad_token_id)[:, :self.model_max_length]
        labels = pad_sequence([sample["labels"] for sample in instances], batch_first=True,
                              padding_value=IGNORE_INDEX)[:, :self.model_max_length]
        pixel_values = [sample["pixel_values"] for sample in instances]
        assert all(value is not None for value in pixel_values), "Invalid VLA Example with `pixel_values = None`!"
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {key: torch.stack([value[key] for value in pixel_values]) for key in pixel_values[0]}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        output = dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels,
                      attention_mask=input_ids.ne(self.pad_token_id))
        for key in CAMERA_KEYS:
            output[key] = [sample[key] for sample in instances]
        return output
