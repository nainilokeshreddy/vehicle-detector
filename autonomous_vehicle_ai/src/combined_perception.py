"""
Combined perception pipeline for the autonomous vehicle project.

Integrates:
- YOLO object detection
- OpenCV lane detection
- Traffic sign classification
"""

from pathlib import Path
import time

import cv2

from src.object_detection import ObjectDetector
from src.lane_detection import LaneDetector
from src.traffic_sign_detection import TrafficSignClassifier


class CombinedPerception:
    """Combine object, lane and traffic-sign perception."""

    def __init__(self):
        """Initialize all perception modules."""

        self.object_detector = ObjectDetector(
            confidence_threshold=0.25
        )

        self.lane_detector = LaneDetector()

        self.traffic_classifier = TrafficSignClassifier()

        # Load the trained traffic-sign model if available.
        model_path = Path("models/traffic_sign_mobilenet.pth")

        if model_path.exists():
            try:
                self.traffic_classifier.load_model(str(model_path))
                self.traffic_model_loaded = True
            except Exception as exc:
                print(f"Warning: traffic-sign model could not be loaded: {exc}")
                self.traffic_model_loaded = False
        else:
            print("Warning: traffic-sign model not found.")
            self.traffic_model_loaded = False

        self.device = getattr(
            self.object_detector,
            "device",
            "cpu"
        )

    def process_image(
        self,
        image_path: str,
        output_dir: str = "outputs/images"
    ) -> dict:
        """
        Process one road image using all perception modules.

        Returns:
            Dictionary containing output path and perception results.
        """

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        output = image.copy()

        # ---------------------------------------------------------
        # 1. Object detection
        # ---------------------------------------------------------
        object_detections, _ = self.object_detector.detect(output)

        output = self.object_detector.draw_annotations(
            output,
            object_detections,
            show_hud=True
        )

        # ---------------------------------------------------------
        # 2. Lane detection
        # ---------------------------------------------------------
        lane_data = self.lane_detector.detect_lanes(output)

        output = self.lane_detector.draw_lane_overlay(
            output,
            lane_data
        )

        # ---------------------------------------------------------
        # 3. Traffic-sign detection/classification
        # ---------------------------------------------------------
        traffic_signs = []

        if self.traffic_model_loaded:
            try:
                traffic_signs = (
                    self.traffic_classifier
                    .detect_and_classify_with_yolo(
                        output,
                        self.object_detector,
                        confidence_threshold=0.25
                    )
                )
            except Exception as exc:
                print(
                    f"Warning: traffic-sign processing failed: {exc}"
                )

        # ---------------------------------------------------------
        # 4. Draw traffic-sign results
        # ---------------------------------------------------------
        for sign in traffic_signs:
            try:
                box = sign.get("box")

                if box is None:
                    continue

                x1, y1, x2, y2 = map(int, box)

                label = sign.get(
                    "class_name",
                    "traffic sign"
                )

                confidence = sign.get(
                    "confidence",
                    0.0
                )

                text = f"{label} {confidence:.1%}"

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    output,
                    text,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

            except Exception:
                continue

        # ---------------------------------------------------------
        # 5. Combined HUD
        # ---------------------------------------------------------
        left_lane_detected = lane_data["left_lane"] is not None
        right_lane_detected = lane_data["right_lane"] is not None

        cv2.rectangle(
            output,
            (10, 10),
            (430, 125),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            output,
            f"Objects: {len(object_detections)}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Left Lane: {'Detected' if left_lane_detected else 'Not detected'}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Right Lane: {'Detected' if right_lane_detected else 'Not detected'}",
            (20, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Traffic Signs: {len(traffic_signs)}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ---------------------------------------------------------
        # 6. Save output
        # ---------------------------------------------------------
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        input_name = Path(image_path).stem

        save_path = output_path / (
            f"{input_name}_combined.jpg"
        )

        cv2.imwrite(str(save_path), output)

        return {
            "output_path": str(save_path),
            "object_count": len(object_detections),
            "left_lane_detected": left_lane_detected,
            "right_lane_detected": right_lane_detected,
            "traffic_signs": traffic_signs,
            "device": str(self.device),
        }

    def process_video(
        self,
        video_path: str,
        output_dir: str = "outputs/videos"
    ) -> dict:
        """
        Process a road video using combined perception.
        """

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise FileNotFoundError(
                f"Could not open video: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        input_name = Path(video_path).stem

        save_path = output_path / (
            f"{input_name}_combined.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(save_path),
            fourcc,
            fps,
            (width, height)
        )

        start_time = time.time()
        processed_frames = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Object detection
                detections, _ = self.object_detector.detect(
                    frame
                )

                frame = self.object_detector.draw_annotations(
                    frame,
                    detections,
                    show_hud=True
                )

                # Lane detection
                lane_data = self.lane_detector.detect_lanes(
                    frame
                )

                frame = self.lane_detector.draw_lane_overlay(
                    frame,
                    lane_data
                )

                # Traffic signs
                traffic_signs = []

                if self.traffic_model_loaded:
                    try:
                        traffic_signs = (
                            self.traffic_classifier
                            .detect_and_classify_with_yolo(
                                frame,
                                self.object_detector,
                                confidence_threshold=0.25
                            )
                        )
                    except Exception:
                        traffic_signs = []

                # Draw traffic signs
                for sign in traffic_signs:
                    try:
                        box = sign.get("box")

                        if box is None:
                            continue

                        x1, y1, x2, y2 = map(
                            int,
                            box
                        )

                        label = sign.get(
                            "class_name",
                            "traffic sign"
                        )

                        confidence = sign.get(
                            "confidence",
                            0.0
                        )

                        text = (
                            f"{label} "
                            f"{confidence:.1%}"
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            text,
                            (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2
                        )

                    except Exception:
                        continue

                # Combined status
                left_lane_detected = lane_data["left_lane"] is not None
                right_lane_detected = lane_data["right_lane"] is not None

                cv2.rectangle(
                    frame,
                    (10, 10),
                    (430, 125),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    f"Objects: {len(detections)}",
                    (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Left Lane: {'Detected' if left_lane_detected else 'Not detected'}",
                    (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Right Lane: {'Detected' if right_lane_detected else 'Not detected'}",
                    (20, 96),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Traffic Signs: {len(traffic_signs)}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                writer.write(frame)

                processed_frames += 1

                if (
                    processed_frames % 10 == 0
                    or processed_frames == total_frames
                ):
                    percentage = (
                        processed_frames / total_frames * 100
                        if total_frames > 0
                        else 0
                    )

                    print(
                        f"[Frame {processed_frames}/"
                        f"{total_frames}] "
                        f"({percentage:.1f}%)"
                    )

        finally:
            cap.release()
            writer.release()

        elapsed = time.time() - start_time

        return {
            "output_path": str(save_path),
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "fps": fps,
            "elapsed_seconds": elapsed,
        }