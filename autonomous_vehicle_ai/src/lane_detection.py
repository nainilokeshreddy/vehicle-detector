"""
Lane detection module for the Autonomous Vehicle Perception System.

Uses classical OpenCV computer vision:
- Grayscale conversion
- Gaussian blur
- Canny edge detection
- Region of interest
- Probabilistic Hough Line Transform
- Left/right lane fitting
- Lane polygon overlay
"""

import os
import time
import cv2
import numpy as np


class LaneDetector:
    """Detect road lane markings using OpenCV."""

    def __init__(
        self,
        canny_low=50,
        canny_high=150,
        hough_threshold=30,
        min_line_length=40,
        max_line_gap=120,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap

    def _region_of_interest(self, edges):
        """Keep the lower road region where lane markings are expected."""
        height, width = edges.shape[:2]

        mask = np.zeros_like(edges)

        polygon = np.array(
            [[
                (int(width * 0.05), height),
                (int(width * 0.42), int(height * 0.55)),
                (int(width * 0.58), int(height * 0.55)),
                (int(width * 0.95), height),
            ]],
            dtype=np.int32,
        )

        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _make_line_points(self, slope, intercept, height, width):
        """Convert a fitted line into two image points."""
        if abs(slope) < 1e-6:
            return None

        y1 = height
        y2 = int(height * 0.58)

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        x1 = max(-width, min(2 * width, x1))
        x2 = max(-width, min(2 * width, x2))

        return (x1, y1, x2, y2)

    def _average_lane_line(self, lines, side, height, width):
        """Fit a single lane line from Hough line candidates."""
        slopes = []
        intercepts = []
        weights = []

        for line in lines:
            x1, y1, x2, y2 = line

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            # Ignore nearly horizontal lines.
            if abs(slope) < 0.3:
                continue

            # Left lane should normally have negative slope.
            if side == "left" and slope >= 0:
                continue

            # Right lane should normally have positive slope.
            if side == "right" and slope <= 0:
                continue

            intercept = y1 - slope * x1
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            slopes.append(slope)
            intercepts.append(intercept)
            weights.append(length)

        if not slopes:
            return None

        slope = np.average(slopes, weights=weights)
        intercept = np.average(intercepts, weights=weights)

        return self._make_line_points(slope, intercept, height, width)

    def detect_lanes(self, image):
        """
        Detect lane lines from an input image.

        Returns a dictionary containing:
        - left_lane
        - right_lane
        - edges
        - roi_edges
        """
        if image is None:
            raise ValueError("Input image is None.")

        if len(image.shape) != 3:
            raise ValueError("Input image must be a color image.")

        height, width = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(
            blurred,
            self.canny_low,
            self.canny_high,
        )

        roi_edges = self._region_of_interest(edges)

        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        candidates = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                candidates.append((x1, y1, x2, y2))

        left_lane = self._average_lane_line(
            candidates,
            "left",
            height,
            width,
        )

        right_lane = self._average_lane_line(
            candidates,
            "right",
            height,
            width,
        )

        return {
            "left_lane": left_lane,
            "right_lane": right_lane,
            "edges": edges,
            "roi_edges": roi_edges,
        }

    def draw_lane_overlay(self, image, lane_data):
        """Draw detected lanes and the drivable lane region."""
        output = image.copy()

        height, width = output.shape[:2]

        left_lane = lane_data.get("left_lane")
        right_lane = lane_data.get("right_lane")

        # Draw individual lane lines.
        if left_lane is not None:
            x1, y1, x2, y2 = left_lane
            cv2.line(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                8,
            )

        if right_lane is not None:
            x1, y1, x2, y2 = right_lane
            cv2.line(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                8,
            )

        # Create filled lane polygon when both sides are available.
        if left_lane is not None and right_lane is not None:
            lx1, ly1, lx2, ly2 = left_lane
            rx1, ry1, rx2, ry2 = right_lane

            polygon = np.array(
                [[
                    (lx1, ly1),
                    (lx2, ly2),
                    (rx2, ry2),
                    (rx1, ry1),
                ]],
                dtype=np.int32,
            )

            overlay = output.copy()
            cv2.fillPoly(
                overlay,
                polygon,
                (0, 255, 0),
            )

            output = cv2.addWeighted(
                output,
                0.75,
                overlay,
                0.25,
                0,
            )

        # Status information.
        left_status = "Detected" if left_lane is not None else "Not detected"
        right_status = "Detected" if right_lane is not None else "Not detected"

        cv2.rectangle(
            output,
            (20, 20),
            (390, 95),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            output,
            f"Left Lane: {left_status}",
            (35, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            output,
            f"Right Lane: {right_status}",
            (35, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return output

    def process_image(self, image_path, output_path=None):
        """Process one road image and save the lane-detected result."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        lane_data = self.detect_lanes(image)
        result = self.draw_lane_overlay(image, lane_data)

        if output_path is None:
            os.makedirs("outputs/images", exist_ok=True)

            name = os.path.splitext(
                os.path.basename(image_path)
            )[0]

            output_path = os.path.join(
                "outputs/images",
                f"{name}_lanes.jpg",
            )

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if not cv2.imwrite(output_path, result):
            raise IOError(
                f"Could not save output image: {output_path}"
            )

        return {
            "output_path": output_path,
            "left_lane_detected": lane_data["left_lane"] is not None,
            "right_lane_detected": lane_data["right_lane"] is not None,
        }

    def process_video(self, input_path, output_path=None):
        """Process a road video frame-by-frame for lane detection."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"Video not found: {input_path}"
            )

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise IOError(
                f"Could not open video: {input_path}"
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30.0

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if output_path is None:
            os.makedirs("outputs/videos", exist_ok=True)

            name = os.path.splitext(
                os.path.basename(input_path)
            )[0]

            output_path = os.path.join(
                "outputs/videos",
                f"{name}_lanes.mp4",
            )

        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            cap.release()
            raise IOError(
                f"Could not create output video: {output_path}"
            )

        processed_frames = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                lane_data = self.detect_lanes(frame)
                annotated = self.draw_lane_overlay(
                    frame,
                    lane_data,
                )

                writer.write(annotated)

                processed_frames += 1

                if (
                    processed_frames % 30 == 0
                    or processed_frames == total_frames
                ):
                    elapsed = time.time() - start_time
                    processing_fps = (
                        processed_frames / elapsed
                        if elapsed > 0
                        else 0
                    )

                    if total_frames > 0:
                        progress = (
                            processed_frames
                            / total_frames
                            * 100
                        )
                        print(
                            f"[Frame {processed_frames}/{total_frames}] "
                            f"({progress:.1f}%) | "
                            f"Speed: {processing_fps:.2f} FPS"
                        )
                    else:
                        print(
                            f"[Frame {processed_frames}] | "
                            f"Speed: {processing_fps:.2f} FPS"
                        )

        finally:
            cap.release()
            writer.release()

        elapsed = time.time() - start_time

        return {
            "output_path": output_path,
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "fps": fps,
            "elapsed_seconds": elapsed,
            "average_processing_fps": (
                processed_frames / elapsed
                if elapsed > 0
                else 0
            ),
        }


if __name__ == "__main__":
    print("Lane detection module loaded successfully.")
