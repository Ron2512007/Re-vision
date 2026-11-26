"""Prepare UIUC's labeled car/no-car images for binary classification."""

import csv
import hashlib
import io
from pathlib import Path
import random
import tarfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "uiuc"
SOURCE = "https://cogcomp.seas.upenn.edu/Data/Car/CarData.tar.gz"
SHA256 = "d5769bff063ad77c933dab50d3a238886c8a0a3fda509d9b0fb0434d8c901480"
SEED = 42


def main():
    cache = ROOT / ".demo-assets" / "CarData.tar.gz"
    if cache.is_file():
        content = cache.read_bytes()
    else:
        with urlopen(SOURCE, timeout=45) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("The UIUC archive differs from the verified download.")
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(content)

    examples = {"car": [], "no_car": []}
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if not member.isfile() or path.parent.as_posix() != "CarData/TrainImages":
                continue
            label = "car" if path.name.startswith("pos-") else "no_car"
            if path.suffix == ".pgm":
                examples[label].append((path.name, archive.extractfile(member).read()))

    if {label: len(images) for label, images in examples.items()} != {"car": 550, "no_car": 500}:
        raise ValueError("The archive has unexpected car/no-car image counts.")

    rows = []
    for label, images in examples.items():
        images.sort(key=lambda item: item[0])
        random.Random(SEED).shuffle(images)
        boundary = len(images) * 4 // 5
        folder = DATA / label
        folder.mkdir(parents=True, exist_ok=True)
        for index, (name, image) in enumerate(images):
            (folder / name).write_bytes(image)
            rows.append({
                "file": f"{label}/{name}",
                "label": label,
                "split": "train" if index < boundary else "test",
            })

    with (DATA / "labels.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["file", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print("Ready: 550 car images + 500 no-car images.")
    print("Fixed 80/20 split: 840 training images, 210 test images.")
    print("Images and labels are in data/uiuc.")


if __name__ == "__main__":
    main()
