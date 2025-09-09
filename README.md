Overview

This repo implements two end-to-end ML tasks on the same dataset:

Regression — predict median house value (MedHouseVal).

Binary classification — predict whether a district is high-value (1) or low-value (0), using the median of MedHouseVal as the threshold.

Both tasks share identical preprocessing and train/test splits to enable apples-to-apples comparisons. Models are built with scikit-learn Pipelines for clarity and reproducibility.
