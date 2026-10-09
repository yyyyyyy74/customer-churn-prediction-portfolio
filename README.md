# Customer Churn Prediction & Retention Analysis

## Project Overview

This project uses machine learning to analyze and predict customer churn in a telecommunications sample dataset. It covers data cleaning, exploratory data analysis (EDA), preprocessing, model comparison, cross-validation, hyperparameter tuning, evaluation, and model interpretation.

**Task:** Binary classification of customer churn (`No = 0`, `Yes = 1`). This is an educational portfolio project based on IBM's publicly available sample dataset, not proprietary customer records.

## Dataset

- **Source:** [IBM Telco Customer Churn sample dataset](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)
- **Rows:** 7,043 customers
- **Original columns:** 21, including `customerID` and the target `Churn`
- **Model inputs:** 19 original feature columns, expanded to 45 after one-hot encoding
- **Target:** `Churn` (`No = 0`, `Yes = 1`)

**The original CSV is not included in this repository.** Download it directly from IBM and save it locally as `data/Telco-Customer-Churn.csv`, as explained below.

## Technologies

- **Python**, **pandas**, and **NumPy** for data processing
- **Matplotlib** for data visualization
- **scikit-learn** for preprocessing, classification, cross-validation, and model evaluation
- **Jupyter Notebook** and **VS Code** for development

## Methodology

1. **Data cleaning:** Checked missing values and duplicates. Replaced 11 blank `TotalCharges` entries (all with zero tenure) with 0 and converted the column to numeric.
2. **Exploratory data analysis:** Investigated churn rates across contract types, services, and payment methods, along with tenure, charges, and correlations.
3. **Train/test split:** Used a stratified 80/20 split (5,634 training customers and 1,409 test customers; `random_state=42`).
4. **Preprocessing:** Applied one-hot encoding to categorical features and standardization to numerical features. Used scikit-learn pipelines for cross-validation so preprocessing is fitted within each training fold.
5. **Model comparison:** Compared Logistic Regression, Decision Tree, and Random Forest using five-fold cross-validation and the churn-class F1-score.
6. **Hyperparameter tuning:** Tuned Logistic Regression's `C` and `class_weight` using `GridSearchCV` on the training data.
7. **Model interpretation:** Examined Logistic Regression coefficients to explore associations between features and predicted churn, without claiming causal effects.

## Results

### Initial Model Comparison (5-Fold Cross-Validation)

| Model | Mean CV F1 (churn class) |
|---|---:|
| Logistic Regression | 0.598 |
| Random Forest | 0.576 |
| Decision Tree | 0.560 |

Logistic Regression achieved the highest initial cross-validation F1-score under the tested settings. Grid search selected `C=0.1` and `class_weight="balanced"`, with a best cross-validation F1-score of **0.633**.

### Tuned Logistic Regression (Test Set)

| Metric | Score |
|---|---:|
| Accuracy | 74.2% |
| Precision (churn) | 50.9% |
| Recall (churn) | 78.6% |
| F1-score (churn) | 61.8% |

The tuned model correctly identified **294 of 374** customers who actually churned in the test set. Compared with the untuned Logistic Regression model, false negatives decreased from **165 to 80**, while false positives increased from **109 to 284**. This illustrates the trade-off between identifying more at-risk customers and making additional false-positive predictions.

**Evaluation note:** Cross-validation scores were used for model selection, while final metrics were computed on the held-out test split. The untuned baseline's test results were inspected earlier in this exploratory project, so the test set was not completely untouched throughout the analysis. Small numerical differences may occur across library versions.

## Key Findings

- Customers with month-to-month contracts had a churn rate of **42.71%**, compared with **11.27%** for one-year contracts and **2.83%** for two-year contracts.
- Customers who churned generally had shorter tenure and higher average monthly charges (**$74.44** versus **$61.27** for customers who stayed).
- Higher churn rates were observed among customers using fiber-optic internet or electronic checks and among those without online security or technical support.
- `tenure` and `TotalCharges` showed a strong positive correlation (**r = 0.826**), which is important when interpreting correlated model features.

These findings describe associations in this dataset; they do not establish causation.

## Repository Structure

```text
customer-churn-prediction-portfolio/
├── notebooks/
│   └── churn_analysis.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The CSV is downloaded into a local `data/` directory and excluded from Git tracking. Local notebook backups and temporary files are also ignored.

## Run the Analysis

The project was tested with **Python 3.13.5** and the dependencies specified in `requirements.txt`.

### 1. Get the Repository

Clone this repository using the HTTPS URL from the green **Code** button on GitHub, or download the repository as a ZIP. Open the resulting project folder in a terminal.

### 2. Create an Environment and Install Dependencies

On macOS or Linux, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 3. Download the Dataset

The original dataset is hosted by IBM and is not redistributed in this repository. From the **project root**, run:

```bash
mkdir -p data
curl -fL https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv -o data/Telco-Customer-Churn.csv
```

### 4. Run the Notebook

Open `notebooks/churn_analysis.ipynb` in VS Code (with the Python and Jupyter extensions), select the `.venv` kernel, and click **Run All**. The notebook reads the CSV using the path `../data/Telco-Customer-Churn.csv` relative to the `notebooks/` directory.

Alternatively, from the project root:

```bash
cd notebooks
jupyter lab churn_analysis.ipynb
```

## Limitations and Future Work

- IBM's example dataset is intended for demonstration; results do not prove performance in real-world deployment.
- Churn is imbalanced, so accuracy alone is insufficient. Recall, false positives, and potential intervention costs should be considered together.
- While cross-validation refits preprocessing within each training fold, additional evaluation on an independent dataset would provide stronger evidence of generalization.
- Future work could explore decision-threshold optimization, cost-sensitive evaluation, and model interpretation with permutation importance or SHAP.

## Data Attribution

The analysis uses the [IBM Telco Customer Churn sample dataset](https://github.com/IBM/telco-customer-churn-on-icp4d). The original CSV is **not redistributed** in this repository; users are directed to download it from IBM.

This is an independent educational portfolio project. It does not claim ownership of the underlying dataset or endorsement by IBM.
