# Decision Tree Algorithms: ID3, C4.5 and CART

A practical demonstration of three classic decision tree algorithms, each implemented **from scratch** in Python (NumPy and pandas) and applied to a diabetes prediction dataset. The results are compared with scikit-learn's decision tree as a reference.

## Objective

To understand how ID3, C4.5 and CART choose their splits, build trees, and perform on the same classification problem (predicting whether a patient has diabetes).

## Algorithms at a glance

| | ID3 | C4.5 | CART |
|---|---|---|---|
| Split criterion | Information Gain (entropy) | Gain Ratio (entropy) | Gini impurity |
| Feature types | Categorical only (numeric features are binned) | Categorical and numeric | Categorical and numeric |
| Split type | Multiway (one branch per value) | Binary (`feature <= threshold`) | Binary (`feature <= threshold`) |
| Extra rule | None | Only features with above-average gain compete on gain ratio | None |

## Project structure

```
Decision Tree/
├── decision_trees_demo.py    # main script (all three algorithms)
├── diabetes_dataset.csv      # dataset (must be in the same folder)
├── cart_tree_sklearn.png     # generated: scikit-learn tree diagram
└── README.md
```

## Dataset

Diabetes prediction dataset (100,000 records).

| Column | Type | Description |
|---|---|---|
| gender | text | Female / Male / Other |
| age | numeric | Age in years |
| hypertension | 0/1 | Has hypertension |
| heart_disease | 0/1 | Has heart disease |
| smoking_history | text | Smoking category |
| bmi | numeric | Body mass index |
| HbA1c_level | numeric | Average blood sugar level |
| blood_glucose_level | numeric | Blood glucose |
| **diabetes** | 0/1 | **Target** (1 = diabetic) |

Note: the classes are imbalanced (roughly 91% non-diabetic), so accuracy alone is misleading. Precision, recall and F1 for the diabetic class are reported too.

## Requirements

- Python 3.9 or newer
- numpy, pandas, scikit-learn, matplotlib

Install:

```
python -m pip install numpy pandas scikit-learn matplotlib
```

On a slow connection:

```
python -m pip install numpy pandas scikit-learn matplotlib --default-timeout=1000 --retries 10
```

## How to run

1. Put `decision_trees_demo.py` and the dataset CSV in the same folder.
2. Open the script and make sure `FILE_NAME` matches your CSV name:
   ```python
   FILE_NAME = "diabetes_dataset.csv"
   ```
3. Run:
   ```
   python decision_trees_demo.py
   ```

## What the script does

1. **Load and inspect** the data and detect the target column.
2. **Split** into 75% training and 25% testing (stratified, `random_state=42`).
3. **Preprocess**
   - ID3: numeric columns with many values are binned into quartiles; text columns are used as categories.
   - C4.5 and CART: text columns are one-hot encoded; numeric columns are used as they are.
4. **Train** ID3, C4.5 and CART from scratch.
5. **Evaluate** each on the test set: accuracy, precision, recall, F1 and a confusion matrix.
6. **Print** each tree as readable if/else rules.
7. **Compare** with scikit-learn's `DecisionTreeClassifier` (entropy and gini) and save a tree diagram.

## Settings you can change

| Setting | Default | Meaning |
|---|---|---|
| `MAX_DEPTH` | 4 | Maximum tree depth (keeps trees readable, limits overfitting) |
| `MAX_THRESHOLDS` | 32 | Candidate split points tried per numeric feature |
| `test_size` | 0.25 | Fraction of data used for testing |

## Results

Fill in after running the script.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| ID3 (information gain) | | | | |
| C4.5 (gain ratio) | | | | |
| CART (gini) | | | | |
| sklearn (entropy) | | | | |
| sklearn (gini) | | | | |

## Limitations

- No pruning is implemented. Overfitting is controlled with a depth limit and a minimum leaf size. The original C4.5 uses pessimistic pruning.
- Numeric splits use up to 32 quantile thresholds per feature, which approximates the exact search for speed.
- Class imbalance is not corrected (no resampling or class weights), so recall for the diabetic class may be low.
- ID3 loses some information because numeric features are binned.

## Possible extensions

- Add pruning (pre-pruning or cost-complexity)
- Handle class imbalance with class weights or resampling
- Plot accuracy against tree depth
- Use k-fold cross-validation for more reliable results