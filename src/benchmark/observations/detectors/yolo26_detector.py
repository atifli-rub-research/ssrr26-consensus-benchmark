from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from benchmark.observations.core.interfaces import Detector


class Yolo26Detector(Detector):
    """Run YOLO inference and return benchmark detection dictionaries."""

    def __init__(self, model_path: str | Path, *, drone_id: str | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            context = f" for {drone_id}" if drone_id is not None else ""
            raise FileNotFoundError(f"YOLO model not found{context}: {self.model_path}")

        self.model = YOLO(str(self.model_path))

    def predict(self, image: Image.Image) -> list[dict]:
        result = self.model.predict(source=image, verbose=False)[0]

        detections = []
        if result.boxes is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, confidence, class_id in zip(boxes_xyxy, confidences, classes):
                detections.append({
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                    "bbox_xyxy": [
                        float(box[0]), float(box[1]),
                        float(box[2]), float(box[3]),
                    ],
                })

        return detections
