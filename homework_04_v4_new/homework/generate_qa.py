import json
from pathlib import Path
import fire
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt  # Added for visualization

# Constants matching the starter code
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Red for track boundaries
    3: (0, 0, 255),  # Blue for track elements
    4: (255, 255, 0),  # Yellow for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Cyan for special elements
}

ORIGINAL_DIMS = (600, 400)  # (width, height)
MIN_BOX_SIZE = 5  # Minimum pixel size for valid karts


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """Extract frame ID and view index from filename"""
    filename = Path(image_path).name
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0


def draw_detections(image_path: str, info_path: str) -> np.ndarray:
    """Visualize detections on image"""
    pil_image = Image.open(image_path)
    draw = ImageDraw.Draw(pil_image)

    with open(info_path) as f:
        info = json.load(f)

    _, view_index = extract_frame_info(image_path)

    if view_index >= len(info["detections"]):
        return np.array(pil_image)

    scale_x = pil_image.width / ORIGINAL_DIMS[0]
    scale_y = pil_image.height / ORIGINAL_DIMS[1]

    for detection in info["detections"][view_index]:
        class_id, track_id, x1, y1, x2, y2 = map(int, detection)
        if class_id != 1:
            continue

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
            continue

        color = (255, 0, 0) if track_id == 0 else COLORS.get(class_id, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    return np.array(pil_image)


def extract_kart_objects(info_path: str, view_index: int) -> list:
    """Extract and validate kart objects with centers"""
    with open(info_path) as f:
        info = json.load(f)

    if view_index >= len(info["detections"]):
        return []

    image_center = (ORIGINAL_DIMS[0] / 2, ORIGINAL_DIMS[1] / 2)
    karts = []

    for detection in info["detections"][view_index]:
        class_id, track_id, x1, y1, x2, y2 = map(int, detection)

        # Filter non-karts and invalid boxes
        if class_id != 1 or (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
            continue

        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        distance = np.linalg.norm(np.array(center) - np.array(image_center))
        karts.append({
            "track_id": track_id,
            "center": center,
            "distance": distance
        })

    # Identify ego kart (closest to center)
    if karts:
        min_dist = min(k["distance"] for k in karts)
        for k in karts:
            k["is_ego"] = (k["distance"] == min_dist)

    return karts


def generate_qa_pairs(info_path: str, view_index: int) -> list:
    """Generate all required QA pairs for one image view"""
    karts = extract_kart_objects(info_path, view_index)
    track_name = json.load(open(info_path)).get("track_name", "unknown")
    qa_pairs = []
    ego_kart = next((k for k in karts if k.get("is_ego")), None)

    # 1. Basic questions
    qa_pairs.extend([
        {"question": "What track is this?", "answer": track_name},
        {"question": "How many karts are there?", "answer": str(len(karts))}
    ])

    # 2. Ego kart question
    if ego_kart:
        qa_pairs.append({
            "question": "Which kart is the ego kart?",
            "answer": f"Kart {ego_kart['track_id']}"
        })

    # 3. Spatial relationships
    if ego_kart:
        for kart in karts:
            if kart["track_id"] == ego_kart["track_id"]:
                continue

            # Calculate positions (strict coordinate comparisons)
            x_pos = "left" if kart["center"][0] < ego_kart["center"][0] else "right"
            y_pos = "front" if kart["center"][1] < ego_kart["center"][1] else "back"

            qa_pairs.extend([
                {
                    "question": f"Is kart {kart['track_id']} to the left or right of the ego kart?",
                    "answer": x_pos
                },
                {
                    "question": f"Is kart {kart['track_id']} in front of or behind the ego kart?",
                    "answer": y_pos
                },
                # Advanced combined position (for extra credit)
                {
                    "question": f"Where is kart {kart['track_id']} relative to the ego kart?",
                    "answer": f"{y_pos} and {x_pos}"
                }
            ])

    # 4. Counting by position
    if ego_kart:
        left = sum(1 for k in karts if k["center"][0] < ego_kart["center"][0])
        front = sum(1 for k in karts if k["center"][1] < ego_kart["center"][1])
        qa_pairs.extend([
            {"question": "How many karts are to the left of the ego kart?", "answer": str(left)},
            {"question": "How many karts are in front of the ego kart?", "answer": str(front)}
        ])

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """Debugging function to visualize and check QA pairs"""
    info_file = Path(info_file)
    image_path = str(info_file.parent / info_file.name.replace("_info.json", f"_{view_index:02d}_im.jpg"))

    # Visualize detections
    print(f"Visualizing: {image_path}")
    annotated_image = draw_detections(image_path, str(info_file))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()

    # Show generated QA pairs
    qa_pairs = generate_qa_pairs(str(info_file), view_index)
    print("\nGenerated QA Pairs:")
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_all(data_dir: str, output_file: str):
    """Generate QA pairs for all images in directory"""
    dataset = []
    data_dir = Path(data_dir)
    output_file = Path(output_file)

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating QA pairs from {data_dir}...")

    for info_file in data_dir.glob("*_info.json"):
        for view_idx in range(4):  # 4 views per frame
            image_path = str(info_file).replace("_info.json", f"_{view_idx:02d}_im.jpg")
            qa_pairs = generate_qa_pairs(str(info_file), view_idx)

            for qa in qa_pairs:
                dataset.append({
                    "image_path": image_path,
                    "question": qa["question"],
                    "answer": qa["answer"]
                })

    print(f"Saving {len(dataset)} QA pairs to {output_file}")
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    fire.Fire({
        "check": check_qa_pairs,
        "generate": generate_all
    })