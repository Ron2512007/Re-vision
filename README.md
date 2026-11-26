# Re-vision
An Intelligent Computer Vision System for Vehicle and Pedestrian Detection Using Selective Image-Region Processing.


This project classifies each input image as **Car** or **No car**.

The downloaded dataset has 550 car images and 500 no-car images. They are grayscale, 100×40-pixel crops. Car examples show side views. The source labels are preserved in `data/uiuc/labels.csv`, alongside a fixed 80/20 train/test split. [UIUC car dataset](https://cogcomp.seas.upenn.edu/Data/Car/)

## This week's experiment

A HOG feature extractor and linear SVM were trained locally on 840 images. It classified **209 of 210** held-out crops correctly.

| Actual label | Predicted Car | Predicted No car |
| --- | ---: | ---: |
| Car | 109 | 1 |
| No car | 0 | 100 |

Open [the sample results](results/dataset_preview.png) to see examples from both classes, including the missed car.

This is a seed-42 random split of the source training crops. The script checks for exact and mirrored duplicates across the split. Related source frames may still be similar. Full-photo performance needs testing.

> I found a labeled car/no-car dataset, prepared a reproducible split, and trained a first classifier. It got 209 of 210 held-out crop examples right. The next experiment is testing on ordinary photos.

## Run

From the project folder in PowerShell, predict one example of each class.

```powershell
.\.venv\Scripts\python.exe car_classifier.py predict data/uiuc/car/pos-100.pgm
.\.venv\Scripts\python.exe car_classifier.py predict data/uiuc/no_car/neg-100.pgm
```

Replace the image path to try another photo. The script prints `Car` or `No car`.

## Rebuild

Use Python 3.13 and run these commands from this folder.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
py -3.13 setup_demo.py
.\.venv\Scripts\python.exe car_classifier.py train
```

The trained model and measured results are saved in `results/car_classifier.xml` and `results/classification_metrics.json`. Prediction works offline.

