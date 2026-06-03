# Requirements

# Functional Requirements

## FR-001

The system shall detect individuals in surveillance videos.

## FR-002

The system shall track individuals across frames.

## FR-003

The system shall estimate trajectories.

## FR-004

The system shall model temporal behavior.

## FR-005

The system shall classify behavioral states.

Classes:

- Normal
- Observation
- Approach
- Following
- Interception
- Crime

## FR-006

The system shall estimate threat probability.

Output:

0.0 → 1.0

## FR-007

The system shall generate visual explanations.

## FR-008

The system shall export experiment results.

---

# Dataset Requirements

## DR-001

Support UCF-Crime.

## DR-002

Support XD-Violence.

## DR-003

Support Urban Threat BR.

---

# Experiment Requirements

## ER-001

Evaluate Precision.

## ER-002

Evaluate Recall.

## ER-003

Evaluate F1 Score.

## ER-004

Evaluate ROC-AUC.

## ER-005

Evaluate Early Prediction Time.

---

# Non-Functional Requirements

## NFR-001

GPU acceleration support.

## NFR-002

Reproducible experiments.

## NFR-003

Version-controlled datasets.

## NFR-004

Experiment tracking.

## NFR-005

Complete documentation.

---

# Infrastructure Requirements

## Recommended Local Machine

CPU:

12+ Cores

RAM:

32GB+

GPU:

RTX 4070+

Preferred:

RTX 4090

---

## Cloud Options

- Google Colab Pro
- RunPod
- Lambda Labs

---

# Success Criteria

## Minimum Goal

F1 Score > 0.80

## Target Goal

F1 Score > 0.90

## Early Prediction

At least 3 seconds before event occurrence.

## Explainability

Attention maps consistent with expert analysis.

---

# Reproducibility

Every experiment must contain:

- Dataset Version
- Model Version
- Hyperparameters
- Random Seed
- Metrics
- Training Logs