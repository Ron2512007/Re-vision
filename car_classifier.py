"""Train a car/no-car classifier and classify individual images."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "uiuc"
RESULTS = ROOT / "results"
MODEL = RESULTS / "car_classifier.xml"
HOG = cv2.HOGDescriptor((96, 40), (16, 16), (8, 8), (8, 8), 9)


def read_image(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not open image {path.name}.")
    return image


def features(image):
    return HOG.compute(cv2.resize(image, (96, 40))).ravel()


def train():
    with (DATA / "labels.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    vectors, labels, splits, fingerprints = [], [], [], {}
    for row in rows:
        image = read_image(DATA / row["file"])
        # A duplicate image, including its mirror, must share the same split.
        fingerprint = min(
            hashlib.sha256(image.tobytes()).digest(),
            hashlib.sha256(cv2.flip(image, 1).tobytes()).digest(),
        )
        if fingerprint in fingerprints and fingerprints[fingerprint] != row["split"]:
            raise ValueError("Duplicate images occur in both training and test data.")
        fingerprints[fingerprint] = row["split"]
        vectors.append(features(image))
        labels.append(int(row["label"] == "car"))
        splits.append(row["split"])

    vectors = np.asarray(vectors, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    training = np.asarray(splits) == "train"
    testing = np.asarray(splits) == "test"
    classifier = cv2.ml.SVM_create()
    classifier.setType(cv2.ml.SVM_C_SVC)
    classifier.setKernel(cv2.ml.SVM_LINEAR)
    classifier.setC(1.0)
    classifier.train(vectors[training], cv2.ml.ROW_SAMPLE, labels[training])
    _, prediction = classifier.predict(vectors[testing])
    predicted = prediction.ravel().astype(int)
    truth = labels[testing]
    confusion = np.zeros((2, 2), dtype=int)
    np.add.at(confusion, (truth, predicted), 1)
    true_negative, false_positive, false_negative, true_positive = confusion.ravel().tolist()
    metrics = {
        "training_images": int(training.sum()),
        "test_images": int(testing.sum()),
        "accuracy": float(np.mean(predicted == truth)),
        "car_precision": true_positive / max(1, true_positive + false_positive),
        "car_recall": true_positive / max(1, true_positive + false_negative),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }
    RESULTS.mkdir(exist_ok=True)
    # Memory serialization supports Windows folders with non-ASCII names.
    storage = cv2.FileStorage("", cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_MEMORY)
    classifier.write(storage, "svm")
    MODEL.write_text(storage.releaseAndGetString(), encoding="utf-8")
    (RESULTS / "classification_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Trained with {training.sum()} images, tested on {testing.sum()} held-out images.")
    print(f"Accuracy {metrics['accuracy']:.1%} | Car precision {metrics['car_precision']:.1%} | Car recall {metrics['car_recall']:.1%}")
    print(f"Cars found {true_positive}, cars missed {false_negative}, false car labels {false_positive}, no-car images correct {true_negative}.")
    print("Saved results/car_classifier.xml and results/classification_metrics.json.")


def predict(photo):
    storage = cv2.FileStorage(MODEL.read_text(encoding="utf-8"), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    classifier = cv2.ml.SVM_create()
    classifier.read(storage.getNode("svm"))
    storage.release()
    _, prediction = classifier.predict(features(read_image(photo)).reshape(1, -1))
    print("Car" if prediction[0, 0] == 1 else "No car")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="Train and evaluate using the prepared dataset")
    prediction = commands.add_parser("predict", help="Print Car or No car for an image")
    prediction.add_argument("photo", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "train":
            train()
        else:
            predict(args.photo)
    except (OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
