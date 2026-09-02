"""
sam_engine.py
SAM2 (Segment Anything 2) wrapper via Hugging Face `transformers`, box-prompted
using Grounding DINO's output boxes. Falls back to original SAM (SamModel/
SamProcessor) if your installed transformers version doesn't expose Sam2
classes yet.
"""

import numpy as np
import torch
from PIL import Image as PILImage

try:
    from transformers import Sam2Model, Sam2Processor
    _SAM_VARIANT = "sam2"
except ImportError:
    from transformers import SamModel, SamProcessor
    _SAM_VARIANT = "sam"


class SamEngine:
    def __init__(self, model_id: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if _SAM_VARIANT == "sam2":
            self.model_id = model_id or "facebook/sam2.1-hiera-large"
        else:
            self.model_id = model_id or "facebook/sam-vit-huge"
        self.processor = None
        self.model = None

    def load(self):
        if _SAM_VARIANT == "sam2":
            self.processor = Sam2Processor.from_pretrained(self.model_id)
            self.model = Sam2Model.from_pretrained(self.model_id).to(self.device).eval()
        else:
            self.processor = SamProcessor.from_pretrained(self.model_id)
            self.model = SamModel.from_pretrained(self.model_id).to(self.device).eval()
        print(f"[SamEngine] loaded '{self.model_id}' (variant: {_SAM_VARIANT})")

    def infer(self, image_rgb: np.ndarray, boxes_xyxy):
        """
        image_rgb  : HxWx3 uint8 RGB image
        boxes_xyxy : list of [x1,y1,x2,y2]

        returns:
            list[np.ndarray(bool)]  one mask per box
        """
        if len(boxes_xyxy) == 0:
            return []

        pil_image = PILImage.fromarray(image_rgb)

        inputs = self.processor(
            images=pil_image,
            input_boxes=[boxes_xyxy],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                multimask_output=False,
            )

        # SAM2's processor output doesn't include "reshaped_input_sizes" at
        # all (its resize/pad pipeline differs from original SAM1's), while
        # SAM1's processor requires it for correct mask cropping. Handle
        # both without hardcoding one signature.
        original_sizes = inputs["original_sizes"].cpu()
        if "reshaped_input_sizes" in inputs:
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                original_sizes,
                inputs["reshaped_input_sizes"].cpu(),
            )[0]
        else:
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                original_sizes,
            )[0]

        # masks shape: [num_boxes, 1, H, W] since multimask_output=False
        results = []
        for i in range(masks.shape[0]):
            mask = masks[i, 0].numpy().astype(bool)
            results.append(mask)

        return results