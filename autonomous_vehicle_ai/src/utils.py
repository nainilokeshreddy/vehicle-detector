"""
Utility Functions for Autonomous Vehicle Perception System.

Responsibilities:
- Visualization helpers (bounding box rendering, HUD display, color palettes).
- File/directory management and output saving helpers.
- Format conversions between RGB, BGR, and tensor representations.
"""


def draw_bounding_boxes(image, detections, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes and class labels on an image.

    Args:
        image (numpy.ndarray): Input image.
        detections (list): Detection results containing coordinates, labels, scores.
        color (tuple): Box color in BGR format.
        thickness (int): Line thickness.

    Returns:
        annotated_image (numpy.ndarray): Image with rendered bounding boxes.
    """
    # Placeholder: Drawing helpers will be implemented in subsequent steps
    return image


def save_processed_image(image, output_path):
    """
    Save processed image frame to the output directory.

    Args:
        image (numpy.ndarray): Processed image frame.
        output_path (str): Destination file path.
    """
    # Placeholder: Image saving utility will be implemented in subsequent steps
    pass
