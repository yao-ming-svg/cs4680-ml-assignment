## Overview
This repo trains two tasks on the same dataset:

1. **Regression** — predict `MedHouseVal` (median house value).  
2. **Binary classification** — predict `is_high_value` (1 if above the median value, else 0).

A shared train/test split and unified preprocessing enable apples-to-apples comparisons. Pipelines prevent leakage and keep code tidy.

---
