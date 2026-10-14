"""
Video Processing Module for Autonomous Vehicle Perception System.

==============================================================================
ACADEMIC & VIVA CONCEPTS EXPLAINED:
==============================================================================
1. Frame-by-Frame Processing:
   A digital video is fundamentally a chronological sequence of static visual
   frames displayed at high speed. Video processing decompresses and processes
   each individual frame (as a NumPy matrix / OpenCV BGR image), executes the
   deep learning perception model on that matrix, annotates the detections,
   and streams the result to an output video encoder.

2. Object Detection in Video Streams:
   Running object detection across consecutive video frames enables an autonomous
   vehicle to track dynamic entities (other vehicles, cyclists, pedestrians) over
   time. Frame-by-frame bounding boxes provide the spatial localization necessary
   for downstream tracking, collision estimation, and trajectory forecasting.

3. Video Capture (cv2.VideoCapture):
   OpenCV's `cv2.VideoCapture` interface manages the video decoder pipeline.
   It accesses video files or live camera hardware, extracts stream metadata
   (resolution, frame rate, total frame count), and sequentially reads frames
   into memory via the `.read()` method.

4. Video Writer (cv2.VideoWriter & FourCC):
   OpenCV's `cv2.VideoWriter` encodes raw processed frames back into a compressed
   video container (such as MP4). The FourCC (Four-Character Code, e.g., 'mp4v')
   specifies the video codec used to compress the frames.

5. FPS (Frames Per Second):
   The frequency at which sequential frames are recorded or displayed.
   Preserving the original FPS during output encoding ensures playback occurs
   at real-world speed without artificial fast-forwarding or slow-motion.

6. MPS (Metal Performance Shaders) vs. CPU:
   Processing 30 frames every second requires high throughput. On Apple Silicon,
   PyTorch leverages MPS to accelerate YOLO matrix multiplications on the M-series
   GPU, maintaining significantly higher inference frames-per-second than CPU
   execution.
==============================================================================
"""

from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from src.object_detection import ObjectDetector, get_device


class VideoProcessor:
    """
    Handles video ingestion, frame-by-frame perception inference, progress tracking,
    and encoded video output saving.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "outputs/videos",
        detector: Optional[ObjectDetector] = None,
    ):
        """
        Initialize the VideoProcessor.

        Args:
            output_dir: Directory where processed videos will be saved.
            detector: Existing ObjectDetector instance. If None, a default one will be created.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detector = detector

    def _open_video_capture(self, video_path: Path) -> Tuple[cv2.VideoCapture, Dict[str, Any]]:
        """
        Open input video and extract essential stream properties.

        Args:
            video_path: Path to input video file.

        Returns:
            cap: Initialized cv2.VideoCapture object.
            meta: Dictionary containing width, height, fps, and total_frames.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"[VideoProcessor] Input video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"[VideoProcessor] Failed to open video stream: {video_path}")

        # Extract stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Guard against corrupted or invalid FPS metadata
        if fps <= 0 or np.isnan(fps):
            print("[VideoProcessor] Warning: Invalid FPS detected in video header. Defaulting to 30.0 FPS.")
            fps = 30.0

        meta = {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
        }
        return cap, meta

    def _create_video_writer(
        self,
        output_path: Path,
        fps: float,
        frame_size: Tuple[int, int],
    ) -> cv2.VideoWriter:
        """
        Initialize the cv2.VideoWriter with a compatible MP4 codec.

        Args:
            output_path: Destination MP4 file path.
            fps: Frame rate for the output video.
            frame_size: (width, height) tuple in pixels.

        Returns:
            out: Initialized cv2.VideoWriter object.
        """
        # 'mp4v' FourCC codec provides broad cross-platform compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

        if not out.isOpened():
            # Fallback codec attempt
            fourcc_fallback = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(str(output_path), fourcc_fallback, fps, frame_size)
            if not out.isOpened():
                raise IOError(f"[VideoProcessor] Could not initialize video writer at: {output_path}")

        return out

    def process_video(
        self,
        input_path: Union[str, Path],
        frame_callback: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
        custom_output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process every video frame sequentially through a perception pipeline callback.

        Args:
            input_path: Path to the input video file.
            frame_callback: Callable taking (frame, frame_idx) and returning annotated_frame.
                            If None, applies the instance's ObjectDetector.
            custom_output_name: Optional custom filename for output video.

        Returns:
            summary (dict): Processing statistics including frames processed, time, and output path.
        """
        input_file = Path(input_path)
        cap, meta = self._open_video_capture(input_file)

        # Resolve output destination
        if custom_output_name:
            out_filename = custom_output_name
        else:
            out_filename = f"{input_file.stem}_detected.mp4"

        output_file = self.output_dir / out_filename
        out = self._create_video_writer(output_file, meta["fps"], (meta["width"], meta["height"]))

        # Default to object detection if no custom callback supplied
        if frame_callback is None:
            if self.detector is None:
                self.detector = ObjectDetector()

            def default_detection_callback(frame: np.ndarray, _idx: int) -> np.ndarray:
                detections, _ = self.detector.detect(frame)
                return self.detector.draw_annotations(frame, detections)

            frame_callback = default_detection_callback

        print("\n" + "=" * 60)
        print("  Starting Video Processing Pipeline")
        print("=" * 60)
        print(f"Input Video      : {input_file}")
        print(f"Resolution       : {meta['width']} x {meta['height']}")
        print(f"Frame Rate (FPS) : {meta['fps']:.2f}")
        print(f"Total Frames     : {meta['total_frames']}")
        print(f"Device           : {get_device().upper()}")
        print(f"Saving Output To : {output_file}")
        print("-" * 60)

        processed_frames = 0
        total_objects_counted = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # End of video stream reached
                    break

                # Apply perception pipeline to current frame
                annotated_frame = frame_callback(frame, processed_frames)

                # Write processed frame to output container
                out.write(annotated_frame)
                processed_frames += 1

                # Display progress updates every 10 frames or on the first frame
                if processed_frames == 1 or processed_frames % 10 == 0 or processed_frames == meta["total_frames"]:
                    elapsed = time.time() - start_time
                    processing_fps = processed_frames / elapsed if elapsed > 0 else 0.0
                    progress_pct = (
                        (processed_frames / meta["total_frames"] * 100)
                        if meta["total_frames"] > 0
                        else 0.0
                    )
                    print(
                        f"[Frame {processed_frames:>4}/{meta['total_frames']}] "
                        f"({progress_pct:>5.1f}%) | "
                        f"Speed: {processing_fps:>5.1f} FPS"
                    )

        finally:
            # Correctly release OpenCV hardware and file descriptors
            cap.release()
            out.release()

        total_elapsed = time.time() - start_time
        avg_fps = processed_frames / total_elapsed if total_elapsed > 0 else 0.0

        print("-" * 60)
        print("  Video Processing Completed Successfully")
        print("=" * 60)
        print(f"Total Frames Processed : {processed_frames}")
        print(f"Total Elapsed Time     : {total_elapsed:.2f} seconds")
        print(f"Average Processing Speed: {avg_fps:.2f} FPS")
        print(f"Saved Output Video     : {output_file}")
        print("=" * 60 + "\n")

        return {
            "output_path": str(output_file),
            "processed_frames": processed_frames,
            "total_frames": meta["total_frames"],
            "resolution": (meta["width"], meta["height"]),
            "fps": meta["fps"],
            "elapsed_seconds": total_elapsed,
            "average_fps": avg_fps,
        }


def process_video_objects(
    video_path: Union[str, Path],
    output_dir: Union[str, Path] = "outputs/videos",
    model_path: str = "models/yolov8n.pt",
    conf_threshold: float = 0.25,
    custom_output_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenient, reusable standalone function to run road object detection on a video.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where annotated output video will be saved (default: 'outputs/videos').
        model_path: Pretrained YOLO weights path (default: 'models/yolov8n.pt').
        conf_threshold: Detection confidence threshold (default: 0.25).
        custom_output_name: Optional custom filename for output video.

    Returns:
        summary (dict): Execution metrics and output video path.
    """
    detector = ObjectDetector(
        model_name_or_path=model_path,
        confidence_threshold=conf_threshold,
    )

    processor = VideoProcessor(
        output_dir=output_dir,
        detector=detector,
    )

    return processor.process_video(
        input_path=video_path,
        custom_output_name=custom_output_name,
    )
