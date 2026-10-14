"""
Main Application Interface - AI Autonomous Vehicle Perception System (Stage 8)

This command-line interface provides a single, unified menu to run all
computer vision and deep learning perception modules:
1. Road Object Detection (Image)
2. Road Lane Detection (Image)
3. Traffic Sign Classification (Image)
4. Combined Perception (Image)
5. Road Object Detection (Video)
6. Road Lane Detection (Video)
7. Combined Perception (Video)
8. Exit
"""

from pathlib import Path
import sys
import time
from typing import Optional

import cv2

# Ensure project root is in Python module search path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Import existing perception modules
from src.object_detection import detect_objects
from src.lane_detection import LaneDetector
from src.traffic_sign_detection import TrafficSignClassifier
from src.combined_perception import CombinedPerception
from src.video_processing import process_video_objects


def resolve_file_path(path_str: str) -> Optional[Path]:
    """
    Resolve a user-provided file path, checking both current working directory
    and the project directory.

    Args:
        path_str: String path entered by user or default.

    Returns:
        Resolved Path object if file exists, otherwise None.
    """
    clean_str = path_str.strip().strip("'\"")
    if not clean_str:
        return None

    try:
        p = Path(clean_str).expanduser()
        if p.is_file():
            return p.resolve()

        proj_p = (PROJECT_DIR / clean_str).resolve()
        if proj_p.is_file():
            return proj_p
    except Exception:
        return None

    return None


def get_input_path(prompt_label: str, default_rel_path: str) -> Optional[Path]:
    """
    Prompt user for a file path, offering a default fallback path.
    Validates file existence and handles errors gracefully without crashing.

    Args:
        prompt_label: Descriptive label for the prompt (e.g. 'image', 'video').
        default_rel_path: Default relative path (e.g. 'dataset/images/road_scene.jpg').

    Returns:
        Resolved Path if valid, or None if invalid/cancelled.
    """
    print(f"\nEnter {prompt_label} path [Default: {default_rel_path}]:")
    try:
        user_input = input(">> ").strip().strip("'\"")
    except (EOFError, KeyboardInterrupt):
        print("\n[Notice] Operation cancelled.")
        return None

    chosen_str = user_input if user_input else default_rel_path
    resolved = resolve_file_path(chosen_str)

    if resolved is None:
        print(f"\n[Error] File not found: '{chosen_str}'")
        print("Please check the path and try again.")
        return None

    return resolved


def run_object_detection_image() -> None:
    """Execute road object detection on a single image."""
    image_path = get_input_path("image", "dataset/images/road_scene.jpg")
    if image_path is None:
        return

    print("\n" + "=" * 55)
    print(" Running Road Object Detection (Image)...")
    print("=" * 55)
    print(f"Input image : {image_path}")

    start_time = time.time()
    try:
        output_dir = PROJECT_DIR / "outputs" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = PROJECT_DIR / "models" / "yolov8n.pt"

        result = detect_objects(
            image_path=str(image_path),
            model_path=str(model_path) if model_path.exists() else "models/yolov8n.pt",
            conf_threshold=0.25,
            output_dir=output_dir,
        )
        elapsed = time.time() - start_time

        print("\n=== Object Detection Results ===")
        print(f"Output File Path  : {result['output_path']}")
        print(f"Objects Detected  : {result['total_objects']}")
        print(f"Inference Device  : {result.get('device', 'cpu').upper()}")
        print(f"Processing Time   : {elapsed:.3f} seconds")

        detections = result.get("detections", [])
        if detections:
            print("\nDetected Road Entities:")
            for idx, det in enumerate(detections, 1):
                box = det["box"]
                print(
                    f"  {idx}. {det['class_name']:<15} "
                    f"Confidence: {det['confidence']:.1%}  "
                    f"Box: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]"
                )
        else:
            print("\nNo road objects detected above confidence threshold.")
    except Exception as exc:
        print(f"\n[Error] Failed to process image for object detection: {exc}")


def run_lane_detection_image() -> None:
    """Execute road lane boundary detection on a single image."""
    image_path = get_input_path("image", "dataset/images/road_scene.jpg")
    if image_path is None:
        return

    print("\n" + "=" * 55)
    print(" Running Road Lane Detection (Image)...")
    print("=" * 55)
    print(f"Input image : {image_path}")

    start_time = time.time()
    try:
        output_dir = PROJECT_DIR / "outputs" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        custom_out = output_dir / f"{image_path.stem}_lanes.jpg"

        detector = LaneDetector()
        result = detector.process_image(
            image_path=str(image_path),
            output_path=str(custom_out),
        )
        elapsed = time.time() - start_time

        left_status = "Detected" if result.get("left_lane_detected") else "Not detected"
        right_status = "Detected" if result.get("right_lane_detected") else "Not detected"

        print("\n=== Lane Detection Results ===")
        print(f"Output File Path  : {result['output_path']}")
        print(f"Left Lane Status  : {left_status}")
        print(f"Right Lane Status : {right_status}")
        print(f"Processing Time   : {elapsed:.3f} seconds")
    except Exception as exc:
        print(f"\n[Error] Failed to process image for lane detection: {exc}")


def run_traffic_sign_classification() -> None:
    """Execute traffic sign recognition and classification on an image."""
    image_path = get_input_path("image", "dataset/images/road_scene.jpg")
    if image_path is None:
        return

    print("\n" + "=" * 55)
    print(" Running Traffic Sign Classification...")
    print("=" * 55)
    print(f"Input image : {image_path}")

    start_time = time.time()
    try:
        model_path = PROJECT_DIR / "models" / "traffic_sign_mobilenet.pth"
        classifier = TrafficSignClassifier(
            model_path=str(model_path) if model_path.exists() else "models/traffic_sign_mobilenet.pth"
        )

        if model_path.exists():
            try:
                classifier.load_model(str(model_path))
            except Exception as load_err:
                print(f"[Warning] Could not load model weights: {load_err}")
        else:
            print(f"[Warning] Weights not found at {model_path}. Initializing default architecture...")
            classifier.prepare_model(pretrained=False)

        prediction = classifier.predict_image(str(image_path))
        elapsed = time.time() - start_time

        # Save annotated image
        output_file_str = "N/A"
        raw_img = cv2.imread(str(image_path))
        if raw_img is not None:
            annotated = classifier.draw_prediction(raw_img, prediction)
            output_dir = PROJECT_DIR / "outputs" / "images"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{image_path.stem}_sign.jpg"
            if cv2.imwrite(str(output_file), annotated):
                output_file_str = str(output_file)

        print("\n=== Traffic Sign Classification Results ===")
        print(f"Output File Path       : {output_file_str}")
        print(f"Predicted Sign Class   : {prediction['class_name']}")
        print(f"Class ID               : {prediction['class_id']}")
        print(f"Classification Score   : {prediction['confidence']:.1%}")
        print(f"Inference Device       : {str(classifier.device).upper()}")
        print(f"Processing Time        : {elapsed:.3f} seconds")
    except Exception as exc:
        print(f"\n[Error] Failed to classify traffic sign: {exc}")


def run_combined_perception_image() -> None:
    """Execute combined perception (objects + lanes + signs) on an image."""
    image_path = get_input_path("image", "dataset/images/road_scene.jpg")
    if image_path is None:
        return

    print("\n" + "=" * 55)
    print(" Running Combined Perception (Image)...")
    print("=" * 55)
    print(f"Input image : {image_path}")

    start_time = time.time()
    try:
        output_dir = PROJECT_DIR / "outputs" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        combined = CombinedPerception()
        result = combined.process_image(
            image_path=str(image_path),
            output_dir=str(output_dir),
        )
        elapsed = time.time() - start_time

        left_status = "Detected" if result.get("left_lane_detected") else "Not detected"
        right_status = "Detected" if result.get("right_lane_detected") else "Not detected"
        traffic_signs = result.get("traffic_signs", [])

        print("\n=== Combined Perception Results ===")
        print(f"Output File Path       : {result['output_path']}")
        print(f"Objects Detected       : {result['object_count']}")
        print(f"Left Lane Status       : {left_status}")
        print(f"Right Lane Status      : {right_status}")
        print(f"Traffic Signs Detected : {len(traffic_signs)}")
        print(f"Inference Device       : {str(result.get('device', 'cpu')).upper()}")
        print(f"Processing Time        : {elapsed:.3f} seconds")

        if traffic_signs:
            print("\nDetected Signs Details:")
            for idx, sign in enumerate(traffic_signs, 1):
                sign_info = sign.get("traffic_sign", {})
                s_name = sign_info.get("class_name", sign.get("class_name", "Unknown"))
                s_conf = sign_info.get("confidence", sign.get("confidence", 0.0))
                print(f"  {idx}. {s_name} (Confidence: {s_conf:.1%})")
    except Exception as exc:
        print(f"\n[Error] Failed to run combined perception on image: {exc}")


def run_object_detection_video() -> None:
    """Execute road object detection on a video stream."""
    video_path = get_input_path("video", "dataset/videos/road_video.mp4")
    if video_path is None:
        return

    print("\n" + "=" * 55)
    print(" Running Road Object Detection (Video)...")
    print("=" * 55)
    print(f"Input video : {video_path}")

    try:
        output_dir = PROJECT_DIR / "outputs" / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = PROJECT_DIR / "models" / "yolov8n.pt"

        result = process_video_objects(
            video_path=video_path,
            output_dir=output_dir,
            model_path=str(model_path) if model_path.exists() else "models/yolov8n.pt",
            conf_threshold=0.25,
        )

        print("\n=== Video Object Detection Results ===")
        print(f"Output File Path       : {result['output_path']}")
        print(f"Frames Processed       : {result['processed_frames']} / {result['total_frames']}")
        print(f"Video Resolution       : {result['resolution'][0]} x {result['resolution'][1]}")
        print(f"Video Native FPS       : {result['fps']:.2f}")
        print(f"Elapsed Processing Time: {result['elapsed_seconds']:.2f} seconds")
        print(f"Average Processing FPS : {result['average_fps']:.2f} FPS")
    except Exception as exc:
        print(f"\n[Error] Failed to process video for object detection: {exc}")


def run_lane_detection_video() -> None:
    """Execute road lane boundary detection on a video stream."""
    video_path = get_input_path("video", "dataset/videos/road_video.mp4")
    if video_path is None:
        return

    print("\n" + "=" * 55)
    print(" Running Road Lane Detection (Video)...")
    print("=" * 55)
    print(f"Input video : {video_path}")

    try:
        output_dir = PROJECT_DIR / "outputs" / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        custom_out = output_dir / f"{video_path.stem}_lanes.mp4"

        detector = LaneDetector()
        result = detector.process_video(
            input_path=str(video_path),
            output_path=str(custom_out),
        )

        print("\n=== Video Lane Detection Results ===")
        print(f"Output File Path       : {result['output_path']}")
        print(f"Frames Processed       : {result['processed_frames']} / {result['total_frames']}")
        print(f"Video Native FPS       : {result['fps']:.2f}")
        print(f"Elapsed Processing Time: {result['elapsed_seconds']:.2f} seconds")
        print(f"Average Processing FPS : {result['average_processing_fps']:.2f} FPS")
    except Exception as exc:
        print(f"\n[Error] Failed to process video for lane detection: {exc}")


def run_combined_perception_video() -> None:
    """Execute combined perception (objects + lanes + signs) on a video stream."""
    video_path = get_input_path("video", "dataset/videos/road_video.mp4")
    if video_path is None:
        return

    print("\n" + "=" * 55)
    print(" Running Combined Perception (Video)...")
    print("=" * 55)
    print(f"Input video : {video_path}")

    try:
        output_dir = PROJECT_DIR / "outputs" / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        combined = CombinedPerception()
        result = combined.process_video(
            video_path=str(video_path),
            output_dir=str(output_dir),
        )

        elapsed = result.get("elapsed_seconds", 0.0)
        processed = result.get("processed_frames", 0)
        avg_fps = processed / elapsed if elapsed > 0 else 0.0

        print("\n=== Video Combined Perception Results ===")
        print(f"Output File Path       : {result['output_path']}")
        print(f"Frames Processed       : {processed} / {result.get('total_frames', 'N/A')}")
        print(f"Video Native FPS       : {result.get('fps', 30.0):.2f}")
        print(f"Elapsed Processing Time: {elapsed:.2f} seconds")
        print(f"Average Processing FPS : {avg_fps:.2f} FPS")
    except Exception as exc:
        print(f"\n[Error] Failed to process video for combined perception: {exc}")


def display_menu() -> None:
    """Print the clean command-line application menu."""
    print()
    print("========================================")
    print(" AI AUTONOMOUS VEHICLE PERCEPTION SYSTEM")
    print("========================================")
    print()
    print("1. Object Detection - Image")
    print("2. Lane Detection - Image")
    print("3. Traffic Sign Classification")
    print("4. Combined Perception - Image")
    print("5. Object Detection - Video")
    print("6. Lane Detection - Video")
    print("7. Combined Perception - Video")
    print("8. Exit")
    print()


def main() -> None:
    """Main application loop."""
    while True:
        display_menu()
        try:
            choice = input("Enter your choice (1-8): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting Autonomous Vehicle Perception System. Goodbye!\n")
            break

        if choice == "1":
            run_object_detection_image()
        elif choice == "2":
            run_lane_detection_image()
        elif choice == "3":
            run_traffic_sign_classification()
        elif choice == "4":
            run_combined_perception_image()
        elif choice == "5":
            run_object_detection_video()
        elif choice == "6":
            run_lane_detection_video()
        elif choice == "7":
            run_combined_perception_video()
        elif choice == "8":
            print("\nExiting Autonomous Vehicle Perception System. Goodbye!\n")
            break
        else:
            print(f"\n[Error] Invalid choice '{choice}'. Please enter a number between 1 and 8.")


if __name__ == "__main__":
    main()

