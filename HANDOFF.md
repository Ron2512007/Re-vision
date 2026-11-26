# Re-vision Car and Pedestrian Vision Project Handoff

## Purpose

This project is a local computer-vision experiment. The planned system accepts a still image frame and independently reports whether it contains a car and whether it contains a pedestrian. The intended final demonstration annotates each detected object with a bounding box and class, reports pixel coordinates, shows a yes/no presence result for each class, and measures processing time. The current implementation is an early car-only crop classifier; it does not yet implement pedestrian inference, full-frame object localization, multi-scale scanning, a graphical interface, or the planned timing/evaluation workflow.

## Proposal intent and algorithm

The proposals describe two independently trained binary classifiers: one for cars and one for pedestrians. Each is intended to use Histograms of Oriented Gradients (HOG) as an image descriptor and a linear Support Vector Machine (SVM) as the classifier. Positive and negative examples are class-specific. The planned car starting data is UIUC; the planned pedestrian source is Daimler. The current code uses UIUC-style 100x40 grayscale crops and has implemented only the car classifier.

HOG divides a grayscale image window into local cells, computes image gradients/edge directions, and summarizes their direction and strength in histograms. Those values describe shape/contour patterns while reducing sensitivity to small illumination shifts. A linear SVM learns a separating hyperplane from labeled HOG feature vectors. At inference, the sign of the decision score yields the binary class; the magnitude can be used as a confidence-like ranking score, but it is not a calibrated probability.

### Implemented crop-classification pipeline

1. Load labeled positive car crops and negative non-car crops (including UIUC labels).
2. Convert/read the images as grayscale and extract fixed-size HOG descriptors using the configured OpenCV HOG descriptor.
3. Form feature vectors and train an OpenCV linear SVM using the training subset.
4. Persist the trained SVM in `results/car_classifier.xml` and metrics/split metadata in `results/classification_metrics.json`.
5. For a supplied image, prepare a compatible grayscale crop, compute the same HOG descriptor, classify with the SVM, and print `Car` or `No car`.

The recorded experiment used 840 crops for training and 210 held-out crops, with 209/210 correct: 109/110 cars and 100/100 non-cars. This result is crop-level classification on a random split; it does not establish performance on arbitrary photographs or on bounding-box detection. The overview notes duplicate checking for exact and mirrored examples, while related source frames can still be similar.

### Intended full-frame detection pipeline (proposal, not implemented)

For each class, run a sliding-window detector over a grayscale frame at multiple scales. At each window, compute HOG and obtain the class SVM score. Keep windows above class-specific thresholds, map scaled window coordinates back to original-frame pixels, then apply non-maximum suppression independently by class to discard redundant overlapping detections. Draw retained `(x, y, width, height)` boxes and labels. Presence is `Yes` when at least one retained detection exists for that class, otherwise `No`. Measure image processing duration and report it separately from model initialization.

The car and pedestrian windows may use different aspect ratios (wider for cars and taller for pedestrians). Thresholds and scan settings should be chosen using validation scenes, not the held-out test set.

## Data and evaluation plan

The proposal calls for representative labeled full-frame scenes containing cars only, pedestrians only, both, or neither, and scenes with multiple objects. Split by source scene/recording session so adjacent frames stay in one partition. Train on the training partition, select thresholds and scan settings on validation, and reserve unseen scenes for final testing. For box evaluation, match predictions to ground-truth boxes one-to-one within class at a preselected IoU threshold and report precision, recall, F1, misses, and false detections. Also report joint frame-level presence correctness and median/slower-frame processing times, with computer and image resolution recorded. Numeric speed target remains undecided until a representative benchmark.

## Project files

- `car_classifier.py`: current crop-level car HOG/SVM training and prediction implementation.
- `setup_demo.py`: prepares/downloads demo data as documented by the project.
- `requirements.txt`: Python dependencies.
- `README.md`: current quick start and measured baseline.
- `PROJECT_OVERVIEW.md`: detailed proposed system, algorithm, evaluation, and milestones.
- `proposal_plan.md`: proposal structure and source references.
- `data/`: prepared UIUC crop dataset and labels/split where present.
- `results/`: trained car SVM, metrics, and preview image.
- `.demo-assets/proposals/`: supplied project proposal source PDFs, extracted text, page images, and inspection JSON.
- `.demo-assets/*.tar.gz`: locally saved source dataset archives.

## Setup and use

Use Python 3.13 as recorded in the README. Create a virtual environment and install `requirements.txt`; run `setup_demo.py` if data need to be prepared. Then run `car_classifier.py train` to rebuild the crop model. The README gives example `predict` commands. The checked-in model supports offline predictions. Do not interpret crop prediction as full-frame object detection.

## Guidance for the next LLM

Treat `PROJECT_OVERVIEW.md` and `proposal_plan.md` as the intended system/proposal, and inspect `car_classifier.py`, `README.md`, and `results/classification_metrics.json` for what is actually implemented. Preserve the distinction between the planned two-class detector and the current car-only crop baseline. A sensible next milestone is to inspect and document dataset provenance/license, prepare pedestrian data and labeled full-frame scenes, and implement/evaluate localization with scene-separated splits.
