"""
dino_engine.py
Grounding DINO wrapper using Hugging Face `transformers`.

Kept as its own module (inside the installed `dino` python package, not in
scripts/) so it's importable as `from dino.dino_engine import DinoEngine`
regardless of where the calling script physically lives, and so it can be
unit-tested / used outside ROS entirely if you ever want to.
"""

import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class DinoEngine:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny",
                 box_threshold: float = 0.35, text_threshold: float = 0.25,
                 device: str = None):
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device).eval()

    def infer(self, image_rgb: np.ndarray, prompt: str):
        """
        image_rgb : HxWx3 uint8 numpy array, RGB order.
        prompt    : e.g. "yellow mug . red bottle . green table ."
                    (lowercase phrases, each ending in " . " - this is the
                    format Grounding DINO expects.)

        returns: list of {"label": str, "score": float, "box_xyxy": [x1,y1,x2,y2]}
        """
        if not prompt or not prompt.strip():
            return []

        pil_image = PILImage.fromarray(image_rgb)
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # NOTE: the exact keyword signature of post_process_grounded_object_detection
        # has shifted slightly across transformers versions. If you hit a
        # TypeError here, check `pip show transformers` and the model card for
        # IDEA-Research/grounding-dino-tiny for the current expected call.
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,    #DEBUG
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],  # (height, width)
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            detections.append({
                "label": label,
                "score": float(score),
                "box_xyxy": [x1, y1, x2, y2],
            })
        return detections
