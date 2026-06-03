# Architecture

# High-Level Architecture

Video Stream
↓
Object Detection
↓
Multi Object Tracking
↓
Feature Extraction
↓
Temporal Modeling
↓
Attention Layer
↓
Threat Prediction
↓
Alert Generation

---

# Module 1 — Object Detection

## Objective

Detect people and relevant objects.

## Recommended Model

YOLOv11

## Alternatives

- RT-DETR
- YOLOv10
- YOLO-NAS

## Framework

PyTorch

## Output

- Bounding Boxes
- Confidence Score
- Class Labels

---

# Module 2 — Multi Object Tracking

## Objective

Track individuals across frames.

## Recommended

ByteTrack

## Alternatives

- DeepSORT
- OC-SORT

## Output

- Person ID
- Position
- Timestamp

---

# Module 3 — Feature Extraction

## Spatial Features

- X Position
- Y Position
- Bounding Box Area

## Motion Features

- Velocity
- Acceleration
- Direction

## Social Features

- Distance Between Individuals
- Relative Motion
- Crowd Density

---

# Module 4 — Temporal Modeling

## Baseline

LSTM

## Alternatives

- Bi-LSTM
- GRU
- Temporal CNN

## Input

Temporal sequences generated from tracking data.

---

# Module 5 — Attention Layer

## Recommended

Self-Attention

## Alternatives

- Multi-Head Attention
- Transformer Encoder

## Purpose

Identify the most relevant temporal segments.

---

# Module 6 — Multi-Task Prediction Head

## Task A

Behavior Classification

Classes:

- Normal
- Observation
- Approach
- Following
- Interception
- Crime

## Task B

Threat Prediction

Output:

Probability from 0.0 to 1.0

---

# Explainability Layer

## Techniques

- Grad-CAM
- Attention Maps
- Saliency Maps
- SHAP

---

# Technology Stack

## Programming Language

Python

## Deep Learning

PyTorch

## Computer Vision

OpenCV

## Data Science

- NumPy
- Pandas
- Scikit-Learn

## Experiment Tracking

Weights & Biases

## Annotation

- CVAT
- Roboflow

---

# Future Deployment

- FastAPI
- Docker
- RTSP Streams
- NVIDIA GPU