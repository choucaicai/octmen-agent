"""OpenVLA batch adapter for raw UAV-ON RGBD episodes."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .prompt_llama2 import build_action_prompt
from .step_collator import IGNORE_INDEX


class RawDepthRLDSBatchTransform:
    """Format a raw RGB-D step as an OpenVLA training sample."""

    def __init__(self, action_tokenizer, tokenizer, image_transform, predict_stop_token=True):
        self.action_tokenizer = action_tokenizer
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.predict_stop_token = predict_stop_token

    def __call__(self, step):
        action = np.asarray(step["action"], dtype=np.float32)
        prompt = build_action_prompt(step["instruction"], self.action_tokenizer(action))
        input_ids = torch.tensor(self.tokenizer(prompt, add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pixel_values = self.image_transform(Image.fromarray(np.asarray(step["image"], dtype=np.uint8)))
        return dict(
            pixel_values=pixel_values, input_ids=input_ids, labels=labels,
            camera_params=torch.tensor(np.asarray(step["camera_params"], dtype=np.float32)),
            camera_poses=torch.tensor(np.asarray(step["camera_poses"], dtype=np.float32)),
            camera_depths=torch.tensor(np.asarray(step["camera_depths"], dtype=np.float32)),
        )
