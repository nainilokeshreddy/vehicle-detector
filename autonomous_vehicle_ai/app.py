import os
import time
import shutil
import subprocess

import cv2
import numpy as np
import streamlit as st

from src.object_detection import ObjectDetector
from src.lane_detection import LaneDetector
from src.traffic_sign_detection import TrafficSignClassifier
from src.combined_perception import CombinedPerception


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Autonomous Vehicle Perception",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DIRECTORIES
# =========================================================

IMAGE_OUTPUT_DIR = "outputs/images"
VIDEO_OUTPUT_DIR = "outputs/videos"

os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)


# =========================================================
# PAGE STYLE
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_object_detector():

    return ObjectDetector(
        model_name_or_path="yolov8n.pt",
        confidence_threshold=0.25,
        filter_road_classes=True,
    )


@st.cache_resource
def load_lane_detector():

    return LaneDetector()


@st.cache_resource
def load_traffic_sign_classifier():

    classifier = TrafficSignClassifier(
        num_classes=43,
        model_path="models/traffic_sign_mobilenet.pth",
    )

    classifier.load_model()

    return classifier


@st.cache_resource
def load_combined_perception():

    return CombinedPerception()


# =========================================================
# IMAGE HELPER
# =========================================================

def uploaded_file_to_image(uploaded_file):

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )

    return image


# =========================================================
# VIDEO OUTPUT HELPERS
# =========================================================

def remove_if_exists(path):

    if os.path.isdir(path):

        shutil.rmtree(path)

    elif os.path.isfile(path):

        os.remove(path)


def find_latest_video(directory, exclude=None):

    if not os.path.exists(directory):
        return None

    video_files = []

    for filename in os.listdir(directory):

        if not filename.lower().endswith(
            (".mp4", ".mov", ".avi", ".mkv")
        ):
            continue

        full_path = os.path.join(
            directory,
            filename,
        )

        if exclude and os.path.abspath(
            full_path
        ) == os.path.abspath(exclude):

            continue

        if os.path.isfile(full_path):

            video_files.append(full_path)

    if not video_files:
        return None

    video_files.sort(
        key=os.path.getmtime,
        reverse=True,
    )

    return video_files[0]


def convert_video_for_browser(
    input_path,
    output_path,
):

    """
    Convert an OpenCV-generated MP4V video into
    an H.264 MP4 when FFmpeg is available.

    If FFmpeg is not installed, the original
    OpenCV video is used.
    """

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:

        return input_path

    try:

        command = [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            output_path,
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0:

            if os.path.isfile(output_path):

                return output_path

    except Exception:
        pass

    return input_path


def validate_video_file(path):

    if not path:
        return False

    if not os.path.isfile(path):
        return False

    if os.path.getsize(path) <= 0:
        return False

    return True


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🚗 Autonomous Vehicle"
)

st.sidebar.caption(
    "AI Perception System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🚘 Object Detection",
        "🛣️ Lane Detection",
        "🚦 Traffic Sign Classification",
        "🧠 Combined Perception",
        "🎥 Video Analysis",
    ],
)

st.sidebar.divider()

st.sidebar.info(
    """
    This application runs locally on your Mac.

    Uploaded images and videos are processed
    using the project's local AI and computer
    vision models.
    """
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">'
        '🚗 AI Autonomous Vehicle Perception System'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-based road perception using computer vision and deep learning'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.header(
        "Project Overview"
    )

    st.write(
        """
        This project is an AI-based autonomous vehicle
        perception prototype. It analyzes road scenes
        and extracts useful visual information such as
        vehicles, pedestrians, lanes, and traffic signs.
        """
    )

    st.subheader(
        "Available Modules"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🚘",
            "Object Detection",
        )

        st.caption(
            "YOLO-based road object detection"
        )

    with col2:

        st.metric(
            "🛣️",
            "Lane Detection",
        )

        st.caption(
            "OpenCV-based lane detection"
        )

    with col3:

        st.metric(
            "🚦",
            "Traffic Signs",
        )

        st.caption(
            "43-class GTSRB classification"
        )

    with col4:

        st.metric(
            "🧠",
            "Combined",
        )

        st.caption(
            "Integrated road perception"
        )

    st.divider()

    st.subheader(
        "System Pipeline"
    )

    st.code(
        """
Input Image / Video
        |
        +----> Object Detection
        |
        +----> Lane Detection
        |
        +----> Traffic Sign Classification
        |
        +----> Combined Perception
                    |
                    v
             Annotated Output
        """,
        language="text",
    )

    st.success(
        "The application can process NEW images and videos "
        "provided by the user."
    )

    st.warning(
        "This is an autonomous vehicle perception prototype. "
        "It is not a complete self-driving vehicle controller."
    )


# =========================================================
# OBJECT DETECTION - IMAGE
# =========================================================

elif page == "🚘 Object Detection":

    st.title(
        "🚘 Object Detection"
    )

    st.write(
        "Upload any new road image and the YOLO model "
        "will detect road-related objects."
    )

    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="object_image",
    )

    if uploaded_file is not None:

        image = uploaded_file_to_image(
            uploaded_file
        )

        if image is None:

            st.error(
                "Unable to read the uploaded image."
            )

        else:

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            st.subheader(
                "Input Image"
            )

            st.image(
                image_rgb,
                width="stretch",
            )

            if st.button(
                "🔍 Analyze Image",
                type="primary",
                width="stretch",
            ):

                with st.spinner(
                    "Running YOLO object detection..."
                ):

                    start_time = time.time()

                    try:

                        detector = (
                            load_object_detector()
                        )

                        detections, _ = (
                            detector.detect(
                                image
                            )
                        )

                        annotated = (
                            detector.draw_annotations(
                                image.copy(),
                                detections,
                            )
                        )

                        processing_time = (
                            time.time()
                            - start_time
                        )

                        annotated_rgb = (
                            cv2.cvtColor(
                                annotated,
                                cv2.COLOR_BGR2RGB,
                            )
                        )

                        st.success(
                            "Object detection completed."
                        )

                        st.subheader(
                            "Detection Result"
                        )

                        st.image(
                            annotated_rgb,
                            width="stretch",
                        )

                        st.divider()

                        col1, col2, col3 = (
                            st.columns(3)
                        )

                        with col1:

                            st.metric(
                                "Objects Detected",
                                len(detections),
                            )

                        with col2:

                            st.metric(
                                "Processing Time",
                                f"{processing_time:.2f} sec",
                            )

                        with col3:

                            st.metric(
                                "Device",
                                str(
                                    getattr(
                                        detector,
                                        "device",
                                        "unknown",
                                    )
                                ),
                            )

                        if detections:

                            st.subheader(
                                "Detected Objects"
                            )

                            for detection in detections:

                                if isinstance(
                                    detection,
                                    dict,
                                ):

                                    name = (
                                        detection.get(
                                            "class_name",
                                            detection.get(
                                                "name",
                                                "Unknown",
                                            ),
                                        )
                                    )

                                    confidence = (
                                        detection.get(
                                            "confidence",
                                            detection.get(
                                                "conf",
                                                0,
                                            ),
                                        )
                                    )

                                    st.write(
                                        f"• **{name}** — "
                                        f"{float(confidence):.1%}"
                                    )

                        else:

                            st.info(
                                "No objects were detected."
                            )

                        output_path = os.path.join(
                            IMAGE_OUTPUT_DIR,
                            "web_object_detection.jpg",
                        )

                        cv2.imwrite(
                            output_path,
                            annotated,
                        )

                        st.caption(
                            f"Result saved to: {output_path}"
                        )

                    except Exception as e:

                        st.error(
                            "Object detection failed."
                        )

                        st.exception(e)


# =========================================================
# LANE DETECTION - IMAGE
# =========================================================

elif page == "🛣️ Lane Detection":

    st.title(
        "🛣️ Lane Detection"
    )

    st.write(
        "Upload a road image to detect the left and right lane boundaries."
    )

    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="lane_image",
    )

    if uploaded_file is not None:

        image = uploaded_file_to_image(
            uploaded_file
        )

        if image is None:

            st.error(
                "Unable to read the uploaded image."
            )

        else:

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            st.subheader(
                "Input Image"
            )

            st.image(
                image_rgb,
                width="stretch",
            )

            if st.button(
                "🛣️ Detect Lanes",
                type="primary",
                width="stretch",
            ):

                with st.spinner(
                    "Detecting road lanes..."
                ):

                    start_time = time.time()

                    try:

                        detector = (
                            load_lane_detector()
                        )

                        lane_data = (
                            detector.detect_lanes(
                                image
                            )
                        )

                        annotated = (
                            detector.draw_lane_overlay(
                                image.copy(),
                                lane_data,
                            )
                        )

                        processing_time = (
                            time.time()
                            - start_time
                        )

                        annotated_rgb = (
                            cv2.cvtColor(
                                annotated,
                                cv2.COLOR_BGR2RGB,
                            )
                        )

                        st.success(
                            "Lane detection completed."
                        )

                        st.subheader(
                            "Lane Detection Result"
                        )

                        st.image(
                            annotated_rgb,
                            width="stretch",
                        )

                        left_detected = (
                            lane_data[
                                "left_lane"
                            ]
                            is not None
                        )

                        right_detected = (
                            lane_data[
                                "right_lane"
                            ]
                            is not None
                        )

                        st.divider()

                        col1, col2, col3 = (
                            st.columns(3)
                        )

                        with col1:

                            if left_detected:

                                st.success(
                                    "Left Lane: Detected"
                                )

                            else:

                                st.warning(
                                    "Left Lane: Not Detected"
                                )

                        with col2:

                            if right_detected:

                                st.success(
                                    "Right Lane: Detected"
                                )

                            else:

                                st.warning(
                                    "Right Lane: Not Detected"
                                )

                        with col3:

                            st.metric(
                                "Processing Time",
                                f"{processing_time:.2f} sec",
                            )

                        output_path = os.path.join(
                            IMAGE_OUTPUT_DIR,
                            "web_lane_detection.jpg",
                        )

                        cv2.imwrite(
                            output_path,
                            annotated,
                        )

                        st.caption(
                            f"Result saved to: {output_path}"
                        )

                    except Exception as e:

                        st.error(
                            "Lane detection failed."
                        )

                        st.exception(e)


# =========================================================
# TRAFFIC SIGN CLASSIFICATION
# =========================================================

elif page == "🚦 Traffic Sign Classification":

    st.title(
        "🚦 Traffic Sign Classification"
    )

    st.write(
        """
        Upload a clearly visible traffic-sign image.
        The trained MobileNetV3 model will classify it
        into one of the 43 GTSRB classes.
        """
    )

    st.warning(
        """
        Upload a cropped or clearly visible traffic sign.
        This module performs classification; it does not
        locate tiny signs inside a full road image.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload a traffic sign image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="traffic_sign_image",
    )

    if uploaded_file is not None:

        image = uploaded_file_to_image(
            uploaded_file
        )

        if image is None:

            st.error(
                "Unable to read the uploaded image."
            )

        else:

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            st.subheader(
                "Input Traffic Sign"
            )

            st.image(
                image_rgb,
                width="stretch",
            )

            if st.button(
                "🚦 Classify Traffic Sign",
                type="primary",
                width="stretch",
            ):

                with st.spinner(
                    "Running traffic sign classification..."
                ):

                    start_time = time.time()

                    try:

                        classifier = (
                            load_traffic_sign_classifier()
                        )

                        temp_path = os.path.join(
                            IMAGE_OUTPUT_DIR,
                            "web_traffic_sign_input.jpg",
                        )

                        cv2.imwrite(
                            temp_path,
                            image,
                        )

                        result = (
                            classifier.predict_image(
                                temp_path
                            )
                        )

                        processing_time = (
                            time.time()
                            - start_time
                        )

                        st.success(
                            "Traffic sign classification completed."
                        )

                        st.subheader(
                            "Prediction"
                        )

                        col1, col2, col3 = (
                            st.columns(3)
                        )

                        with col1:

                            st.metric(
                                "Class ID",
                                result[
                                    "class_id"
                                ],
                            )

                        with col2:

                            st.metric(
                                "Predicted Sign",
                                result[
                                    "class_name"
                                ],
                            )

                        with col3:

                            st.metric(
                                "Confidence",
                                f'{result["confidence"]:.1%}',
                            )

                        st.divider()

                        col1, col2 = (
                            st.columns(2)
                        )

                        with col1:

                            st.write(
                                f"**Device:** "
                                f"{classifier.device}"
                            )

                        with col2:

                            st.write(
                                f"**Processing Time:** "
                                f"{processing_time:.2f} sec"
                            )

                    except Exception as e:

                        st.error(
                            "Traffic sign classification failed."
                        )

                        st.exception(e)


# =========================================================
# COMBINED PERCEPTION - IMAGE
# =========================================================

elif page == "🧠 Combined Perception":

    st.title(
        "🧠 Combined Perception"
    )

    st.write(
        """
        Upload a new road image and run the integrated
        object detection, lane detection, and traffic-sign
        perception pipeline.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        key="combined_image",
    )

    if uploaded_file is not None:

        image = uploaded_file_to_image(
            uploaded_file
        )

        if image is None:

            st.error(
                "Unable to read the uploaded image."
            )

        else:

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            st.subheader(
                "Input Image"
            )

            st.image(
                image_rgb,
                width="stretch",
            )

            if st.button(
                "🧠 Run Combined Perception",
                type="primary",
                width="stretch",
            ):

                with st.spinner(
                    "Running combined perception..."
                ):

                    start_time = time.time()

                    try:

                        system = (
                            load_combined_perception()
                        )

                        temp_input_path = os.path.join(
                            IMAGE_OUTPUT_DIR,
                            "web_combined_input.jpg",
                        )

                        cv2.imwrite(
                            temp_input_path,
                            image,
                        )

                        result = (
                            system.process_image(
                                temp_input_path
                            )
                        )

                        processing_time = (
                            time.time()
                            - start_time
                        )

                        output_path = result.get(
                            "output_path"
                        )

                        output_image = None

                        if (
                            output_path
                            and os.path.isfile(
                                output_path
                            )
                        ):

                            output_image = (
                                cv2.imread(
                                    output_path
                                )
                            )

                        st.success(
                            "Combined perception completed."
                        )

                        if output_image is not None:

                            output_rgb = (
                                cv2.cvtColor(
                                    output_image,
                                    cv2.COLOR_BGR2RGB,
                                )
                            )

                            st.subheader(
                                "Combined Result"
                            )

                            st.image(
                                output_rgb,
                                width="stretch",
                            )

                        st.divider()

                        col1, col2, col3, col4 = (
                            st.columns(4)
                        )

                        with col1:

                            st.metric(
                                "Objects",
                                result.get(
                                    "object_count",
                                    0,
                                ),
                            )

                        with col2:

                            st.metric(
                                "Left Lane",
                                (
                                    "Detected"
                                    if result.get(
                                        "left_lane_detected",
                                        False,
                                    )
                                    else "Not Detected"
                                ),
                            )

                        with col3:

                            st.metric(
                                "Right Lane",
                                (
                                    "Detected"
                                    if result.get(
                                        "right_lane_detected",
                                        False,
                                    )
                                    else "Not Detected"
                                ),
                            )

                        with col4:

                            st.metric(
                                "Device",
                                result.get(
                                    "device",
                                    "MPS/CPU",
                                ),
                            )

                        st.write(
                            f"**Processing Time:** "
                            f"{processing_time:.2f} seconds"
                        )

                        if result.get(
                            "traffic_signs"
                        ):

                            st.subheader(
                                "Traffic Sign Results"
                            )

                            for sign in result[
                                "traffic_signs"
                            ]:

                                st.write(
                                    sign
                                )

                        else:

                            st.info(
                                "No traffic-sign classification "
                                "result was produced for this road image."
                            )

                    except Exception as e:

                        st.error(
                            "Combined perception failed."
                        )

                        st.exception(e)


# =========================================================
# VIDEO ANALYSIS
# =========================================================

elif page == "🎥 Video Analysis":

    st.title(
        "🎥 Video Analysis"
    )

    st.write(
        """
        Upload a new road video and process it using
        object detection, lane detection, or combined perception.
        """
    )

    uploaded_video = st.file_uploader(
        "Upload a road video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv",
        ],
        key="road_video",
    )

    video_mode = st.selectbox(
        "Select Processing Mode",
        [
            "Object Detection",
            "Lane Detection",
            "Combined Perception",
        ],
    )

    if uploaded_video is not None:

        input_video_path = os.path.join(
            VIDEO_OUTPUT_DIR,
            "web_uploaded_input.mp4",
        )

        with open(
            input_video_path,
            "wb",
        ) as video_file:

            video_file.write(
                uploaded_video.getbuffer()
            )

        st.success(
            "Video uploaded successfully."
        )

        st.subheader(
            "Input Video"
        )

        st.video(
            uploaded_video.getvalue()
        )

        if st.button(
            "🎬 Process Video",
            type="primary",
            width="stretch",
        ):

            start_time = time.time()

            processed_frames = 0

            actual_output_path = None

            try:

                # =================================================
                # OBJECT DETECTION VIDEO
                # =================================================

                if video_mode == "Object Detection":

                    st.info(
                        "Running YOLO object detection frame-by-frame..."
                    )

                    detector = (
                        load_object_detector()
                    )

                    output_path = os.path.join(
                        VIDEO_OUTPUT_DIR,
                        "web_object_detection_video.mp4",
                    )

                    remove_if_exists(
                        output_path
                    )

                    cap = cv2.VideoCapture(
                        input_video_path
                    )

                    if not cap.isOpened():

                        raise RuntimeError(
                            "Unable to open uploaded video."
                        )

                    fps = cap.get(
                        cv2.CAP_PROP_FPS
                    )

                    if fps <= 0:
                        fps = 30.0

                    width = int(
                        cap.get(
                            cv2.CAP_PROP_FRAME_WIDTH
                        )
                    )

                    height = int(
                        cap.get(
                            cv2.CAP_PROP_FRAME_HEIGHT
                        )
                    )

                    total_frames = int(
                        cap.get(
                            cv2.CAP_PROP_FRAME_COUNT
                        )
                    )

                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(
                            *"mp4v"
                        ),
                        fps,
                        (
                            width,
                            height,
                        ),
                    )

                    if not writer.isOpened():

                        cap.release()

                        raise RuntimeError(
                            "Unable to create output video."
                        )

                    progress = st.progress(
                        0
                    )

                    while True:

                        ret, frame = (
                            cap.read()
                        )

                        if not ret:
                            break

                        detections, _ = (
                            detector.detect(
                                frame
                            )
                        )

                        annotated = (
                            detector.draw_annotations(
                                frame,
                                detections,
                            )
                        )

                        writer.write(
                            annotated
                        )

                        processed_frames += 1

                        if total_frames > 0:

                            progress.progress(
                                min(
                                    processed_frames
                                    / total_frames,
                                    1.0,
                                )
                            )

                    cap.release()
                    writer.release()

                    actual_output_path = (
                        output_path
                    )

                # =================================================
                # LANE DETECTION VIDEO
                # =================================================

                elif video_mode == "Lane Detection":

                    st.info(
                        "Running lane detection frame-by-frame..."
                    )

                    detector = (
                        load_lane_detector()
                    )

                    output_path = os.path.join(
                        VIDEO_OUTPUT_DIR,
                        "web_lane_detection_video.mp4",
                    )

                    remove_if_exists(
                        output_path
                    )

                    result = (
                        detector.process_video(
                            input_video_path,
                            output_path,
                        )
                    )

                    actual_output_path = (
                        output_path
                    )

                    if isinstance(
                        result,
                        dict,
                    ):

                        processed_frames = (
                            result.get(
                                "frames_processed",
                                result.get(
                                    "frame_count",
                                    0,
                                ),
                            )
                        )

                # =================================================
                # COMBINED PERCEPTION VIDEO
                # =================================================

                else:

                    st.info(
                        "Running combined perception video processing..."
                    )

                    # IMPORTANT:
                    # CombinedPerception.process_video()
                    # expects an OUTPUT DIRECTORY,
                    # NOT an output file path.

                    output_dir = (
                        VIDEO_OUTPUT_DIR
                    )

                    expected_output = os.path.join(
                        output_dir,
                        "web_uploaded_input_combined.mp4",
                    )

                    remove_if_exists(
                        expected_output
                    )

                    system = (
                        load_combined_perception()
                    )

                    result = (
                        system.process_video(
                            input_video_path,
                            output_dir,
                        )
                    )

                    # The module itself creates:
                    #
                    # web_uploaded_input_combined.mp4

                    if isinstance(
                        result,
                        dict,
                    ):

                        processed_frames = (
                            result.get(
                                "frames_processed",
                                result.get(
                                    "frame_count",
                                    0,
                                ),
                            )
                        )

                    # Prefer the exact expected filename.
                    if os.path.isfile(
                        expected_output
                    ):

                        actual_output_path = (
                            expected_output
                        )

                    else:

                        actual_output_path = (
                            find_latest_video(
                                output_dir,
                                exclude=input_video_path,
                            )
                        )

                # =================================================
                # VERIFY OUTPUT
                # =================================================

                processing_time = (
                    time.time()
                    - start_time
                )

                if not validate_video_file(
                    actual_output_path
                ):

                    raise RuntimeError(
                        "Video processing finished, "
                        "but a valid output video file "
                        "could not be found."
                    )

                # =================================================
                # BROWSER COMPATIBILITY
                # =================================================

                browser_output_path = os.path.join(
                    VIDEO_OUTPUT_DIR,
                    "web_browser_ready.mp4",
                )

                remove_if_exists(
                    browser_output_path
                )

                browser_video = (
                    convert_video_for_browser(
                        actual_output_path,
                        browser_output_path,
                    )
                )

                if not validate_video_file(
                    browser_video
                ):

                    browser_video = (
                        actual_output_path
                    )

                # =================================================
                # RESULT
                # =================================================

                st.success(
                    "Video processing completed."
                )

                st.subheader(
                    "Processed Video"
                )

                with open(
                    browser_video,
                    "rb",
                ) as video_file:

                    video_bytes = (
                        video_file.read()
                    )

                st.video(
                    video_bytes
                )

                st.divider()

                col1, col2, col3 = (
                    st.columns(3)
                )

                with col1:

                    if processed_frames:

                        st.metric(
                            "Frames Processed",
                            processed_frames,
                        )

                    else:

                        st.metric(
                            "Processing",
                            "Completed",
                        )

                with col2:

                    st.metric(
                        "Processing Time",
                        f"{processing_time:.2f} sec",
                    )

                with col3:

                    file_size = os.path.getsize(
                        browser_video
                    )

                    st.metric(
                        "Output Size",
                        f"{file_size / (1024 * 1024):.2f} MB",
                    )

                st.caption(
                    f"Output saved to: {actual_output_path}"
                )

                if browser_video != actual_output_path:

                    st.caption(
                        "A browser-compatible copy was created "
                        "for video playback."
                    )

            except Exception as e:

                st.error(
                    "Video processing failed."
                )

                st.exception(e)