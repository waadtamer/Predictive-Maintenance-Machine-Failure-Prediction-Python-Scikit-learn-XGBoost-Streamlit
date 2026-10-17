# Predictive-Maintenance-Machine-Failure-Prediction-Python-Scikit-learn-XGBoost-Streamlit
# Predictive Maintenance: Machine Failure Prediction

A machine learning project that predicts whether a machine will fail from its sensor readings, using the AI4I 2020 Predictive Maintenance dataset. The final model (Gradient Boosting) is served through a Streamlit app.


## Project summary

**Problem.** A machine that breaks down unexpectedly is costly, but replacing parts too early wastes money. The goal is to flag machines that are likely to fail so maintenance can be planned in advance.

**Data.** 10,000 machine records with 5 sensor readings and a machine type. Only 3.4% of the rows are failures, so the classes are highly imbalanced.

**Approach.**
1. Explore the data and check quality (duplicates, missing values, outliers).
2. Detect and remove **data leakage** (columns that reveal the answer).
3. Engineer two physically meaningful features (`Temp diff` and `Power`).
4. Compare three models (Gradient Boosting, Random Forest, XGBoost) with the same **5-fold stratified cross-validation**.
5. Pick the best model, save it with its threshold and feature list, and deploy it in a Streamlit app.

**Result.** Gradient Boosting reached Recall 0.75, Precision 0.93 and F1 0.83 in cross-validation. Feature engineering improved F1 from 0.72 to 0.83.

## Dataset
Link'https://www.kaggle.com/kernels/fork-version/310730991'

مهى
It is a synthetic dataset that imitates real industrial data.

| Column | Description | Used as feature |
|---|---|---|
| `Type` | Product quality variant (L / M / H) | Yes (one-hot encoded) |
| `Air temperature [K]` | Ambient temperature | Yes |
| `Process temperature [K]` | Process temperature | Yes |
| `Rotational speed [rpm]` | Spindle speed | Yes |
| `Torque [Nm]` | Applied torque | Yes |
| `Tool wear [min]` | Minutes of tool use | Yes |
| `Machine failure` | Target: 1 if the machine failed | Target |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Failure modes: tool wear, heat dissipation, power, overstrain, random | No (leakage) |
| `UDI`, `Product ID` | Identifiers | No |

## Notebook walkthrough

The notebook follows this order. Each step explains what the code does and what it found.

### 1. Loading and inspecting the data
- Load `ai4i2020.csv` with pandas and look at it with `head()`, `tail()`, `describe()`, `info()` and `dtypes`.
- **Duplicates:** 0. **Missing values:** 0% in every column. No cleaning was needed.

### 2. Outlier check
- Used the IQR rule (values beyond 1.5 x IQR from the quartiles).
- Outliers were found only in `Rotational speed` (418 rows) and `Torque` (69 rows). The temperatures and tool wear have none.
- I kept them: they are plausible sensor readings (high speed, high torque), not data errors, and tree-based models are not sensitive to them.

### 3. Exploratory data analysis
- **Correlation with the target:** a bar chart of how each numeric column correlates with `Machine failure`.
- **Machine Type:** average sensor values per type, plotted as bar charts. Failure rate by type: L 3.92%, M 2.77%, H 2.09%.
- **Ranges:** min, max and median of each sensor, to understand the scale of the data (these ranges are also used for the input limits in the app).
- **Failure modes:** counts of each cause: TWF 46, HDF 115, PWF 95, OSF 98, RNF 19.
- **Tool wear by Type:** a boxplot showing similar tool wear across the three types (mean about 107 to 108 minutes).

### 4. Preparing features and target
- Dropped `UDI`, `Product ID`, the target, and the five failure-mode columns from the inputs.
- Converted `Type` into numbers with one-hot encoding (`drop_first=True`), which gives `Type_L` and `Type_M`. Type `H` is the reference (both equal 0).

### 5. Baseline model (no engineered features)
- Split the data 80/20 and trained a default Gradient Boosting model on the 7 original columns.
- Single-split result: Accuracy 0.984, Precision 0.84, Recall 0.59, F1 0.69.
- Accuracy looks great but Recall shows the model misses about 4 out of 10 real failures, which is why accuracy is not the metric used for decisions.

### 6. Experiment: balancing with `sample_weight`
- Trained with `sample_weight="balanced"` to make the model pay more attention to failures.
- Recall rose to 0.87 but Precision fell to 0.32 (F1 0.47), meaning most alerts would be false alarms.
- Lowering the threshold made it worse: at 0.1, Recall reached 0.95 but Precision dropped to 0.14.
- **Decision:** rejected. I kept the default weighting with a 0.5 threshold.

### 7. Feature engineering
- Added `Temp diff` (process temperature minus air temperature) and `Power` (torque x rotational speed x 2pi / 60).
- Re-split with `stratify=y` so the test set keeps the same failure rate.
- Single-split result with 9 features: Recall 0.78, Precision 0.95, F1 0.85 (confusion matrix: 1929 correct "OK", 3 false alarms, 15 missed failures, 53 caught failures).

### 8. Cross-validation of Gradient Boosting
- A single split can be lucky, so I evaluated with 5-fold `StratifiedKFold` (shuffled, `random_state=42`).
- Result: Recall 0.75, Precision 0.93, F1 0.83. This is the honest estimate and the one used in this README.

### 9. Saving the final model
- Trained a final Gradient Boosting model on **all** the data.
- Saved a dictionary with the model, the threshold (0.5) and the ordered feature list to `gradient_boosting_fe_threshold.pkl`, so the app uses exactly the same setup as the notebook.
- A quick load-and-predict check confirms the file works. Its scores are inflated (the model has seen those rows), so it is a sanity check only, not an evaluation.

### 10. Random Forest
- Same cross-validation: Recall 0.73, Precision 0.95, F1 0.82.
- Single split: Recall 0.75, Precision 0.94, F1 0.84.

### 11. XGBoost
- XGBoost does not accept square brackets in column names, so I made a copy of the data with `[` and `]` removed from the column names (the original names stay untouched for the app).
- Cross-validation: Recall 0.72, Precision 0.88, F1 0.79. Single split: Recall 0.74, Precision 0.89, F1 0.81.
- Saved in the same dictionary format as `xgboost_fe_threshold.pkl`.

### 12. Feature importance
- Read `feature_importances_` from the final Gradient Boosting model and plotted it as a horizontal bar chart (see [Results](#results)).

## Key decisions explained

**Why Recall and Precision instead of accuracy?**
A model that always says "no failure" gets about 96.6% accuracy and catches nothing. Recall measures how many real failures are caught, Precision measures how many alerts are real, and F1 balances the two.

**What was the data leakage?**
`TWF`, `HDF`, `PWF`, `OSF` and `RNF` describe why a machine failed, so they are only known after the failure. Including them gave unrealistic results (Recall about 0.97). I noticed the scores dropped when a column derived from them was removed, investigated, and removed all of them.

**Why cross-validation?**
Single train/test splits depend on which rows land in the test set. With only 339 failures, the scores can be lucky or unlucky. Using the same 5 folds for every model makes the comparison fair. Note that the first baseline split was not stratified, so single-split numbers from different stages of the notebook are not directly comparable. Cross-validation removes that problem.

**Why Gradient Boosting?**
Gradient Boosting and Random Forest are practically tied on F1. Gradient Boosting has the higher Recall, and in maintenance a missed failure costs more than a false alarm.

**Why was the final model trained on all the data?**
After the model is chosen and evaluated with cross-validation, training on all rows gives the deployed model the most information. The cross-validation scores are the performance estimate.

## Results

### Model comparison (5-fold stratified cross-validation)

| Model | Recall | Precision | F1 |
|---|---|---|---|
| Gradient Boosting, no engineered features (baseline) | 0.62 | 0.86 | 0.72 |
| **Gradient Boosting, with engineered features** | **0.75** | **0.93** | **0.83** |
| Random Forest, with engineered features | 0.73 | 0.95 | 0.82 |
| XGBoost, with engineered features | 0.72 | 0.88 | 0.79 |

### Feature importance (Gradient Boosting)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `Power` (engineered) | 31.4% |
| 2 | `Tool wear [min]` | 20.4% |
| 3 | `Temp diff` (engineered) | 19.4% |
| 4 | `Rotational speed [rpm]` | 17.5% |
| 5 | `Torque [Nm]` | 7.7% |
| 6 | `Type_L` | 2.7% |
| 7 | `Process temperature [K]` | 0.7% |
| 8 | `Air temperature [K]` | 0.2% |
| 9 | `Type_M` | 0.1% |

Both engineered features are in the top three. The raw air and process temperatures barely matter on their own, but their difference is the third most important feature.

Importance is shared between correlated features (`Power` is built from `Torque` and `Rotational speed`), so these values show what the model relies on, not what causes failures.

## Streamlit app

The app asks for the machine type and sensor readings, computes `Temp diff` and `Power` exactly like the notebook, and shows the failure probability. If the probability is at or above the saved threshold (0.5), it warns that the machine likely needs maintenance; otherwise it reports the machine as healthy.

## How to run
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure

```
predictive-maintenance/
├── notebook.ipynb                      # EDA, modeling, cross-validation
├── app.py                              # Streamlit app
├── gradient_boosting_fe_threshold.pkl  # model + threshold + feature list
├── requirements.txt
└── README.md


https://github.com/user-attachments/assets/cc18ba6a-79fe-4fbc-b4a9-a998fc7f7d5d

