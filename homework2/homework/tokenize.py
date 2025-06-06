from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .bsq import Tokenizer


def tokenize(tokenizer: Path, output: Path, *images: Path):
    """
    Tokenize images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    output: Path to save the tokenize image tensor.
    images: Path to the image / images to compress.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")  # for Arm Macs
        if len(images) == 1 and Path(images[0]).is_dir():
            images = Path(images[0]).glob("*.jpg")
    else:
        print("GPU not available, using CPU")
        device = torch.device("cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))

    # Load and compress all images
    compressed_tensors = []
    for image_path in tqdm(images):
        image = Image.open(image_path)
        x = torch.tensor(np.array(image), dtype=torch.uint8, device=device)
        with torch.inference_mode():
            x = x.float() / 255.0 - 0.5
            cmp_image = tk_model.encode_index(x)
            compressed_tensors.append(cmp_image.cpu())

    # Store the tensor in the lowest number of bits possible
    compressed_tensor = torch.stack(compressed_tensors)
    # We rely on numpy here for uint support and faster loading (not that this really matters at this size)
    np_compressed_tensor = compressed_tensor.numpy()
    if np_compressed_tensor.max() < 2**8:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint8)
    elif np_compressed_tensor.max() < 2**16:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint16)
    else:
        np_compressed_tensor = np_compressed_tensor.astype(np.uint32)

    torch.save(np_compressed_tensor, output)


if __name__ == "__main__":
    from fire import Fire

    Fire(tokenize)
