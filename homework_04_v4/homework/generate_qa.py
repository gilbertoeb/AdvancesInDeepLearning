import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Updating file

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    """Extract kart objects and identify the center kart."""
    with open(info_path) as f:
        info = json.load(f)

    if view_index >= len(info["detections"]):
        return []

    frame_detections = info["detections"][view_index]
    karts = []
    image_center = (img_width // 2, img_height // 2)

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        if class_id != 1:  # Only process karts
            continue

        # Scale coordinates
        scale_x = img_width / ORIGINAL_WIDTH
        scale_y = img_height / ORIGINAL_HEIGHT
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if too small or out of bounds
        if (x2_scaled - x1_scaled < min_box_size or
                y2_scaled - y1_scaled < min_box_size or
                x2_scaled < 0 or x1_scaled > img_width or
                y2_scaled < 0 or y1_scaled > img_height):
            continue

        # Calculate center point
        center_x = (x1_scaled + x2_scaled) // 2
        center_y = (y1_scaled + y2_scaled) // 2

        # Get kart name
        kart_name = info["karts"][track_id] if track_id < len(info["karts"]) else f"kart_{track_id}"

        karts.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y),
            "is_ego": track_id == 0  # Track_id 0 is the ego kart
        })

    # Find kart closest to center (if no ego kart found)
    if not any(kart["is_ego"] for kart in karts) and karts:
        min_dist = float('inf')
        for kart in karts:
            dist = (kart["center"][0] - image_center[0]) ** 2 + (kart["center"][1] - image_center[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                kart["is_ego"] = True

    return karts

def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    """Extract track information from the info.json file."""
    with open(info_path) as f:
        info = json.load(f)
    return info.get("track", "unknown_track")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    """Generate question-answer pairs for a given view."""
    qa_pairs = []
    info_path = Path(info_path)

    # Get image filename
    base_name = info_path.stem.replace("_info", "")
    image_file = f"{base_name}_{view_index:02d}_im.jpg"
    rel_image_path = f"train/{image_file}"  # Changed from 'valid' to 'train'

    # Get track info
    with open(info_path) as f:
        info = json.load(f)
    track_name = info.get("track", "unknown_track")

    # Get kart information
    karts = []
    frame_detections = info["detections"][view_index]
    image_center = (img_width // 2, img_height // 2)

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        if class_id != 1:  # Only process karts
            continue

        # Scale coordinates
        scale_x = img_width / ORIGINAL_WIDTH
        scale_y = img_height / ORIGINAL_HEIGHT
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if too small or out of bounds
        if (x2_scaled - x1_scaled < 5 or y2_scaled - y1_scaled < 5 or
            x2_scaled < 0 or x1_scaled > img_width or
            y2_scaled < 0 or y1_scaled > img_height):
            continue

        # Calculate center point
        center_x = (x1_scaled + x2_scaled) // 2
        center_y = (y1_scaled + y2_scaled) // 2

        # Get kart name
        kart_name = info["karts"][track_id] if track_id < len(info["karts"]) else f"kart_{track_id}"

        karts.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y),
            "is_ego": track_id == 0
        })

    # Find ego kart (either track_id=0 or closest to center)
    ego_kart = next((k for k in karts if k["is_ego"]), None)
    if not ego_kart and karts:
        min_dist = float('inf')
        for kart in karts:
            dist = (kart["center"][0] - image_center[0])**2 + (kart["center"][1] - image_center[1])**2
            if dist < min_dist:
                min_dist = dist
                ego_kart = kart
                ego_kart["is_ego"] = True

    if not ego_kart:
        return qa_pairs

    # 1. Ego car question
    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego_kart["kart_name"],
        "image_file": rel_image_path
    })

    # 2. Total karts question
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts)),
        "image_file": rel_image_path
    })

    # 3. Track information question
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name,
        "image_file": rel_image_path
    })

    # 4. Relative position questions
    other_karts = [k for k in karts if k["instance_id"] != ego_kart["instance_id"]]
    for kart in other_karts:
        # Left/right determination
        if kart["center"][0] < ego_kart["center"][0]:
            lr_answer = "left"
        else:
            lr_answer = "right"

        # Front/back determination (smaller y is front)
        if kart["center"][1] < ego_kart["center"][1]:
            fb_answer = "front"
        else:
            fb_answer = "back"

        # Individual questions
        qa_pairs.append({
            "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
            "answer": lr_answer,
            "image_file": rel_image_path
        })
        qa_pairs.append({
            "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
            "answer": fb_answer,
            "image_file": rel_image_path
        })
        qa_pairs.append({
            "question": f"Where is {kart['kart_name']} relative to the ego car?",
            "answer": f"{fb_answer} and {lr_answer}",
            "image_file": rel_image_path
        })

    # 5. Counting questions
    left_count = sum(1 for k in karts if k["center"][0] < ego_kart["center"][0])
    right_count = len(karts) - left_count - 1  # exclude ego kart

    front_count = sum(1 for k in karts if k["center"][1] < ego_kart["center"][1])
    back_count = len(karts) - front_count - 1  # exclude ego kart

    qa_pairs.extend([
        {
            "question": "How many karts are to the left of the ego car?",
            "answer": str(left_count),
            "image_file": rel_image_path
        },
        {
            "question": "How many karts are to the right of the ego car?",
            "answer": str(right_count),
            "image_file": rel_image_path
        },
        {
            "question": "How many karts are in front of the ego car?",
            "answer": str(front_count),
            "image_file": rel_image_path
        },
        {
            "question": "How many karts are behind the ego car?",
            "answer": str(back_count),
            "image_file": rel_image_path
        }
    ])

    return qa_pairs


def generate_all(data_dir: str = "../data/train", output_dir: str = "../data/train"):
    """Generate QA pairs for all training data."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for info_file in data_path.glob("*_info.json"):
        frame_id = info_file.stem.split("_")[0]
        qa_pairs = []

        with open(info_file) as f:
            info = json.load(f)
            num_views = len(info["detections"])

        for view_index in range(num_views):
            qa_pairs.extend(generate_qa_pairs(str(info_file), view_index))

        output_file = output_path / f"{frame_id}_qa_pairs.json"
        with open(output_file, "w") as f:
            json.dump(qa_pairs, f, indent=2)

        print(f"Generated {len(qa_pairs)} QA pairs for {info_file.name}")

def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate_all": generate_all
    })


if __name__ == "__main__":
    main()
