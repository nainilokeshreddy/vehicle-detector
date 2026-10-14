"""
Road Object Detection Module for Autonomous Vehicle Perception System.

==============================================================================
ACADEMIC & VIVA CONCEPTS EXPLAINED:
==============================================================================
1. What YOLO Does:
   YOLO ("You Only Look Once") is a state-of-the-art, real-time object detection
   deep learning architecture. Traditional detectors (like R-CNN) evaluate
   thousands of proposed regions sequentially. In contrast, YOLO treats object
   detection as a single regression problem, passing the entire image through
   a convolutional neural network in a single forward pass to predict both
   bounding box coordinates and class probabilities simultaneously.

2. What Inference Means:
   Inference is the phase where a previously trained neural network evaluates
   unseen input data (such as a camera photo) using its fixed weights. Unlike
   training, inference does not calculate gradients or perform backpropagation;
   it only computes the forward pass to produce real-time predictions.

3. Confidence Threshold:
   A probability value between 0.0 and 1.0 representing how certain the model
   is about a detection. Detections with confidence scores below this threshold
   (default: 0.25) are filtered out to prevent false positive detections.

4. Bounding Boxes:
   Rectangular spatial coordinates [x1, y1, x2, y2] identifying the precise
   pixel boundaries enclosing a detected road entity:
   - (x1, y1): Top-left corner of the box
   - (x2, y2): Bottom-right corner of the box

5. Class Labels:
   Categorical identifiers assigned to detected bounding boxes indicating the
   object category (e.g., 'car', 'person', 'truck') based on the dataset
   ontology (such as the 80-class COCO benchmark).

6. MPS (Metal Performance Shaders) versus CPU:
   - MPS: Apple Silicon hardware accelerator using integrated Apple GPU cores
     via macOS Metal framework. Provides fast parallel tensor operations.
   - CPU: Central Processing Unit fallback. Works on any machine but executes
     tensor math sequentially, resulting in lower frames per second (FPS).

7. Output Image:
   An annotated visualization of the input image showing colored bounding boxes,
   class labels, confidence percentages, and a perception HUD banner, saved into
   the project's 'outputs/images/' folder.
==============================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Standard road-relevant categories from the 80 COCO object detection classes
ROAD_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}

# Distinct BGR color mapping for intuitive visual recognition on camera feeds
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "car": (0, 255, 0),            # Bright Green
    "bus": (255, 165, 0),          # Cyan / Light Blue
    "truck": (0, 215, 255),        # Gold / Yellow-Orange
    "motorcycle": (255, 0, 255),   # Magenta
    "bicycle": (0, 255, 255),      # Yellow
    "person": (0, 128, 255),       # Orange
    "traffic light": (0, 0, 255),  # Red
    "stop sign": (0, 0, 255),      # Red
}
DEFAULT_COLOR: Tuple[int, int, int] = (0, 255, 0)


def get_device() -> str:
    """
    Automatically select the optimal computing device for local inference.

    Checks if Apple's MPS (Metal Performance Shaders) is available on macOS.
    If available, returns 'mps' for hardware-accelerated inference; otherwise,
    falls back to 'cpu'.

    Returns:
        device (str): 'mps' or 'cpu'.
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_model_path(model_name_or_path: str = "yolov8n.pt") -> str:
    """
    Resolve model weight path relative to project structure or models/ directory.

    Args:
        model_name_or_path: File name (e.g. 'yolov8n.pt') or path to weights.

    Returns:
        resolved_path (str): Absolute or relative filesystem path to weights.
    """
    candidate = Path(model_name_or_path)
    if candidate.is_file():
        return str(candidate.resolve())

    # Check project-relative models/ directory
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    weights_in_models = models_dir / candidate.name
    if weights_in_models.is_file():
        return str(weights_in_models.resolve())

    # Fallback to models directory path where weights can be placed/downloaded
    return str(weights_in_models)


class ObjectDetector:
    """
    Reusable Road Object Detector class using a pretrained YOLO model.
    """

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        filter_road_classes: bool = True,
    ):
        """
        Initialize the detector, load weights, and configure hardware device.

        Args:
            model_name_or_path: Model weight name or file path (default: 'yolov8n.pt').
            confidence_threshold: Cutoff score to discard weak detections (default: 0.25).
            device: 'mps' or 'cpu'. If None, detected automatically.
            filter_road_classes: If True, keep only road-relevant entities.
        """
        self.confidence_threshold = float(confidence_threshold)
        self.device = device if device else get_device()
        self.filter_road_classes = filter_road_classes
        self.model_path = resolve_model_path(model_name_or_path)

        # Load pretrained YOLO network
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names

    def detect(
        self, image: Union[np.ndarray, str, Path]
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """
        Execute YOLO forward pass on an image and extract structured detections.

        Args:
            image: OpenCV BGR image array or path to image file.

        Returns:
            detections (list of dict): Structured list containing:
                - 'class_name': str
                - 'class_id': int
                - 'confidence': float (0.0 to 1.0)
                - 'box': [x1, y1, x2, y2] (integer pixel coordinates)
            raw_result: The raw Ultralytics Results object.
        """
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        first_result = results[0]
        detections: List[Dict[str, Any]] = []

        if first_result.boxes is not None and len(first_result.boxes) > 0:
            boxes = first_result.boxes.xyxy.cpu().numpy()
            confs = first_result.boxes.conf.cpu().numpy()
            classes = first_result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, classes):
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")

                if self.filter_road_classes and class_name not in ROAD_CLASSES:
                    continue

                detections.append({
                    "class_name": class_name,
                    "class_id": int(cls_id),
                    "confidence": float(conf),
                    "box": [int(coord) for coord in box],
                })

        return detections, first_result

    def draw_annotations(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        show_hud: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, confidence badges, and perception HUD on frame.

        Args:
            image: Input OpenCV BGR image.
            detections: List of detection dictionaries.
            show_hud: If True, renders top perception metrics banner.

        Returns:
            annotated_image: Processed image ready for display or saving.
        """
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            name = det["class_name"]
            conf = det["confidence"]
            color = CLASS_COLORS.get(name, DEFAULT_COLOR)

            # Draw obstacle bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Format label badge: "Car 87.3%"
            badge_text = f"{name.title()} {conf * 100:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(
                badge_text, font, font_scale, thickness
            )

            # Draw filled badge background for high legibility
            badge_top = max(0, y1 - text_h - 6)
            cv2.rectangle(
                annotated,
                (x1, badge_top),
                (x1 + text_w + 6, badge_top + text_h + 6),
                color,
                -1,
            )

            # Render badge text
            cv2.putText(
                annotated,
                badge_text,
                (x1 + 3, badge_top + text_h + 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA,
            )

        if show_hud:
            # Render HUD overlay showing detection metrics
            hud_text = (
                f"Objects: {len(detections)} | "
                f"Device: {self.device.upper()} | "
                f"Conf: {self.confidence_threshold:.2f}"
            )
            cv2.putText(
                annotated,
                hud_text,
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

        return annotated


def detect_objects(
    image_path: Union[str, Path],
    model_path: str = "models/yolov8n.pt",
    conf_threshold: float = 0.25,
    output_dir: Union[str, Path] = "outputs/images",
    custom_output_name: Optional[str] = None,
    filter_road_objects: bool = True,
) -> Dict[str, Any]:
    """
    Convenient, reusable standalone function to detect road objects in an image.

    This function can be directly imported and invoked by subsequent modules
    (e.g., video_processing.py or a combined perception pipeline).

    Args:
        image_path: Filepath to the input road image.
        model_path: Path or filename for pretrained YOLO weights (default: 'models/yolov8n.pt').
        conf_threshold: Confidence score threshold (default: 0.25).
        output_dir: Directory where annotated output will be stored (default: 'outputs/images').
        custom_output_name: Optional custom filename for output image.
        filter_road_objects: If True, retains only road-relevant COCO classes.

    Returns:
        result_dict (dict):
            - 'output_path': Path to the saved annotated image.
            - 'detections': List of detected road objects with boxes & confidences.
            - 'total_objects': Number of detected objects.
            - 'device': Hardware device used ('mps' or 'cpu').
            - 'annotated_image': OpenCV BGR image array with drawn annotations.
    """
    input_file = Path(image_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    # Read image using OpenCV
    image = cv2.imread(str(input_file))
    if image is None:
        raise ValueError(f"OpenCV failed to decode image at: {image_path}")

    # Instantiate detector
    detector = ObjectDetector(
        model_name_or_path=model_path,
        confidence_threshold=conf_threshold,
        filter_road_classes=filter_road_objects,
    )

    # Run inference
    detections, _ = detector.detect(image)

    # Render annotations
    annotated_image = detector.draw_annotations(image, detections)

    # Prepare output directory and destination filepath
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if custom_output_name:
        dest_filename = custom_output_name
    else:
        dest_filename = f"{input_file.stem}_detected{input_file.suffix}"

    final_output_path = out_dir_path / dest_filename

    # Save output image
    write_ok = cv2.imwrite(str(final_output_path), annotated_image)
    if not write_ok:
        raise IOError(f"Could not save annotated output to {final_output_path}")

    return {
        "output_path": str(final_output_path),
        "detections": detections,
        "total_objects": len(detections),
        "device": detector.device,
        "annotated_image": annotated_image,
    }
