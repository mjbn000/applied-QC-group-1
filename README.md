# AQC Group 1 Project

The NSL-KDD dataset used in this repository is sourced from here:

https://www.kaggle.com/datasets/hassan06/nslkdd

**Folders:**
- nsl-kdd: The original dataset, provided as reference
- scripts: All Python code written for the project
- data: Contains the reduced dataset as well as evaluation results

**Steps:**

1. Run preprocessing.py to reduce the provided KDDTrain+ and KDDTest+ subsets.

2. Run baseline.py, which uses the reduced dataset.

3. Run the QSVC and QVC scripts in your preferred order.

4. Evaluate your results, or adjust the subsets and QML scripts as desired. Examples:
- Changing the sample_size parameter in preprocessing.py for a desired sample ratio.
- Selecting a different number of features to reduce the subsets to.
- Increasing the number of reps in both QML scripts.
