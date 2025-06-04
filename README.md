# Deepfake Detection on FaceForensics++ (FF++)

This repository contains the codebase for training, evaluating, and ensembling deepfake detection models using the [FaceForensics++ (FF++)](https://github.com/ondyari/FaceForensics) dataset. The project focuses on binary classification (real vs. fake) using convolutional neural networks (Xception) and includes tools for preprocessing video data, extracting faces, and evaluating models.

## Table of Contents

- [Overview](#overview)
- [Dataset Access](#dataset-access)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Scripts](#scripts)
- [Citation](#citation)
- [Credits](#credits)

---

## Overview

The goal of this project is to accurately detect deepfakes using models trained on the FaceForensics++ dataset. The code supports the full pipeline, from frame extraction and face cropping, to training and ensemble-based evaluation of multiple models specialized for different manipulation techniques.

## Dataset Access

**IMPORTANT:**  
The FaceForensics++ dataset **cannot be downloaded directly**. To access the data, you must fill out a request form as described in the [official FaceForensics++ GitHub repository](https://github.com/ondyari/FaceForensics). After approval, you will receive download instructions and credentials.

## Project Structure
.
├── extract_frames.py # Extracts frames from videos
├── extract_faces.py # Detects and saves faces from frames
├── Train_Binary_Model.py # Trains a binary classifier (Xception)
├── evaluate.py # Evaluates individual models
├── ensemble.py # Ensemble evaluation using multiple models
└── (your data and weights folders, not included)


## How to Use

1. **Request the FF++ dataset:**  
   Follow the instructions [here](https://github.com/ondyari/FaceForensics) to obtain the dataset.

2. **Preprocess videos:**  
   - Use `extract_frames.py` to extract frames from each video.
   - Use `extract_faces.py` to crop faces from each frame (requires [MTCNN](https://github.com/ipazc/mtcnn) and Pillow).

3. **Train models:**  
   - Use `Train_Binary_Model.py` to train an Xception-based classifier on the preprocessed data.

4. **Evaluate models:**  
   - Use `evaluate.py` to generate classification reports for each model.
   - Use `ensemble.py` to perform ensemble evaluation over multiple manipulation types.

## Scripts

### `extract_frames.py`
Extracts frames from each video in a directory and saves them as individual images.

### `extract_faces.py`
Detects and crops faces from the extracted frames, saving each face as a separate image.

### `Train_Binary_Model.py`
Trains an Xception-based binary classifier (real vs. fake) with data augmentation, validation, and logging.

### `evaluate.py`
Evaluates the performance of trained models on test data, outputs classification reports and confusion matrices.

### `ensemble.py`
Combines predictions from multiple trained models (one for each manipulation method) and computes ensemble statistics (mean probability and majority voting).

## Citation

If you use this code or the dataset, please cite the [FaceForensics++ paper](https://arxiv.org/abs/1803.09179) and give proper credit to the dataset authors.

---

## Credits

Developed by:

- **Manuel Muñoz**
- **Aldrin Chavez**


