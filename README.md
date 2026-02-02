# SME Credit Risk Analysis

## Predicting Non-Performing Loans for Ghana's Financial Sector

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Dataset Description](#dataset-description)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Methodology](#methodology)
8. [Models Implemented](#models-implemented)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Results Summary](#results-summary)
11. [Feature Importance](#feature-importance)
12. [Business Recommendations](#business-recommendations)
13. [Future Improvements](#future-improvements)
14. [Contributing](#contributing)
15. [License](#license)

---

## 🎯 Project Overview

This project develops a machine learning solution to predict **Non-Performing Loans (NPLs)** for Small and Medium Enterprises (SMEs) in Ghana's banking sector. The model helps financial institutions make data-driven lending decisions, reducing default rates while maintaining healthy loan portfolios.

### Key Objectives

- Build a classification model to predict loan default probability
- Achieve minimum **80% accuracy** and **65% recall** for NPL detection
- Provide interpretable results for business stakeholders
- Create a deployable model package for production use

---

## 💼 Business Problem

### Context

Ghana's banking sector faces significant challenges with SME loan defaults. High NPL rates (around 20%) lead to:

- **Financial losses** from unrecovered loan amounts
- **Increased provisioning** requirements
- **Reduced lending capacity** for the broader economy
- **Higher interest rates** passed on to borrowers

### Solution Value

A predictive model that accurately identifies high-risk loan applications can:

| Benefit | Impact |
|---------|--------|
| Reduce NPL rate | Target: 20% → 15% |
| Annual savings | Estimated GHS 10.7M |
| Faster decisions | Automated risk scoring |
| Consistent criteria | Objective assessment |

---

## 📊 Dataset Description

### Overview

| Attribute | Value |
|-----------|-------|
| **Records** | ~5,000 loan applications |
| **Features** | 25+ variables |
| **Target** | `loan_status` (Performing / Non-Performing) |
| **Time Period** | Historical loan data |
| **Source** | Synthetic data based on Ghana banking patterns |

### Feature Categories

#### 1. Business Demographics
| Feature | Description | Type |
|---------|-------------|------|
| `sector` | Business industry | Categorical |
| `region` | Geographic location in Ghana | Categorical |
| `years_in_operation` | Business age in years | Numerical |
| `number_of_employees` | Workforce size | Numerical |
| `business_registration_type` | Legal structure | Categorical |

#### 2. Financial Metrics
| Feature | Description | Type |
|---------|-------------|------|
| `annual_revenue_ghs` | Yearly revenue in Ghana Cedis | Numerical |
| `profit_margin_pct` | Net profit percentage | Numerical |
| `current_ratio` | Current assets / Current liabilities | Numerical |
| `debt_to_equity` | Total debt / Shareholder equity | Numerical |
| `cash_flow_ratio` | Operating cash flow ratio | Numerical |

#### 3. Credit History
| Feature | Description | Type |
|---------|-------------|------|
| `credit_bureau_score` | Credit rating (300-850) | Numerical |
| `previous_loan_defaults` | Number of past defaults | Numerical |
| `banking_relationship_years` | Years with current bank | Numerical |
| `existing_loan_count` | Number of active loans | Numerical |

#### 4. Loan Characteristics
| Feature | Description | Type |
|---------|-------------|------|
| `loan_amount_ghs` | Requested loan amount | Numerical |
| `loan_term_months` | Repayment period | Numerical |
| `interest_rate_pct` | Annual interest rate | Numerical |
| `loan_to_revenue_ratio` | Loan amount / Annual revenue | Numerical |
| `collateral_coverage_pct` | Collateral value / Loan amount | Numerical |

#### 5. Owner Information
| Feature | Description | Type |
|---------|-------------|------|
| `owner_age` | Business owner's age | Numerical |
| `owner_education_level` | Highest education attained | Categorical |
| `owner_has_other_businesses` | Multiple business ownership | Binary |

### Target Variable

```
loan_status:
  - "Performing"     → Loan payments on schedule (Label: 0)
  - "Non-Performing" → Loan in default/arrears  (Label: 1)
```

---

## 📁 Project Structure

```
sme-credit-risk-analysis/
│
├── 📓 SME_Credit_Risk_Analysis_Simplified.ipynb  # Main tutorial notebook
├── 📄 README.md                                   # This file
├── 📊 sme_loan_applications_ghana.csv            # Dataset (required)
├── 📋 requirements.txt                            # Python dependencies
│
├── 📂 outputs/                        # Generated after running notebook
│   ├── *.png                          # Visualization plots
│
└── 📂 models/                         # Generated after running notebook
    └── sme_credit_risk_model.pkl      # Saved trained model
```

> **Note:** The `outputs/` and `models/` folders are created automatically when you run the notebook. They will contain all generated visualizations and the trained model.

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/sme-credit-risk-analysis.git
cd sme-credit-risk-analysis
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
```

### Requirements File

Create a `requirements.txt` with:

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
joblib>=1.0.0
jupyter>=1.0.0
```

---

## 🚀 Quick Start

### Option 1: Run the Jupyter Notebook

```bash
jupyter notebook SME_Credit_Risk_Analysis.ipynb
```

Then execute cells sequentially (Shift + Enter).

### Option 2: Run as Python Script

```python
# Quick prediction example
import joblib
import pandas as pd

# Load the trained model
model_package = joblib.load('sme_credit_risk_model.pkl')

# Prepare new application data
new_application = {
    'credit_bureau_score': 650,
    'annual_revenue_ghs': 500000,
    'current_ratio': 1.5,
    'debt_to_equity': 0.8,
    # ... add all required features
}

# Make prediction
# (Remember to preprocess: encode categoricals, scale features)
```

---

## 🔬 Methodology

### Workflow Diagram

```
┌─────────────────┐
│  1. Data Load   │
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. EDA         │ ← Understand distributions, correlations, NPL patterns
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Preprocess  │ ← Encode categories, scale features
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. Split Data  │ ← 80% train, 20% test (stratified)
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. SMOTE       │ ← Balance classes in training set
└────────┬────────┘
         ▼
┌─────────────────┐
│  6. Train       │ ← Logistic Regression, SVM, Naive Bayes
└────────┬────────┘
         ▼
┌─────────────────┐
│  7. Evaluate    │ ← Compare metrics, select champion
└────────┬────────┘
         ▼
┌─────────────────┐
│  8. Deploy      │ ← Save model for production
└─────────────────┘
```

### Key Preprocessing Steps

| Step | Technique | Purpose |
|------|-----------|---------|
| **Encoding** | Label Encoding (binary), One-Hot (multi-class) | Convert text to numbers |
| **Scaling** | StandardScaler (z-score normalization) | Normalize feature ranges |
| **Balancing** | SMOTE (50% minority ratio) | Handle class imbalance |
| **Splitting** | Stratified 80/20 split | Maintain class distribution |

---

## 🤖 Models Implemented

### 1. Logistic Regression

```python
LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',
    penalty='l2',
    C=1.0
)
```

**Strengths:**
- Highly interpretable (coefficient analysis)
- Fast training and prediction
- Works well with linearly separable data
- Provides probability outputs

**Use Case:** Best for understanding feature impact on risk.

---

### 2. Support Vector Machine (SVM)

```python
SVC(
    random_state=42,
    kernel='rbf',
    class_weight='balanced',
    probability=True
)
```

**Strengths:**
- Effective in high-dimensional spaces
- Handles non-linear relationships (RBF kernel)
- Robust to outliers

**Use Case:** Best for complex decision boundaries.

---

### 3. Gaussian Naive Bayes

```python
GaussianNB()
```

**Strengths:**
- Extremely fast training
- Works well with small datasets
- Simple and efficient

**Use Case:** Good baseline model, fast predictions.

---

## 📏 Evaluation Metrics

### Metrics Explained

| Metric | Formula | Interpretation | Target |
|--------|---------|----------------|--------|
| **Accuracy** | (TP+TN) / Total | Overall correctness | ≥ 80% |
| **Precision** | TP / (TP+FP) | When we predict NPL, how often correct? | ≥ 70% |
| **Recall** | TP / (TP+FN) | What % of actual NPLs do we catch? | ≥ 65% |
| **F1-Score** | 2×(P×R)/(P+R) | Balance of precision & recall | - |
| **ROC-AUC** | Area under ROC | Overall discrimination ability | ≥ 0.85 |

### Confusion Matrix Interpretation

```
                    Predicted
                 0 (Performing)    1 (NPL)
              ┌─────────────────┬─────────────────┐
Actual    0   │  True Negative  │ False Positive  │
(Performing)  │    (Correct)    │  (False Alarm)  │
              ├─────────────────┼─────────────────┤
Actual    1   │ False Negative  │  True Positive  │
(NPL)         │ (Missed NPL!)   │    (Correct)    │
              └─────────────────┴─────────────────┘
```

### Business Cost of Errors

| Error Type | Business Impact | Estimated Cost |
|------------|-----------------|----------------|
| **False Negative** (Missed NPL) | Loan defaults, write-off | GHS 75,000 per loan |
| **False Positive** (False alarm) | Lost business opportunity | GHS 9,000 (12% of loan interest) |

---

## 📈 Results Summary

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.85 | 0.72 | 0.68 | 0.70 | 0.89 |
| SVM | 0.84 | 0.70 | 0.65 | 0.67 | 0.87 |
| Naive Bayes | 0.78 | 0.58 | 0.72 | 0.64 | 0.82 |

> **Note:** Actual results will vary based on your dataset.

### Champion Model

🏆 **Logistic Regression** selected based on:
- Highest ROC-AUC score
- Best balance of precision and recall
- High interpretability for stakeholders
- Meets all performance targets

---

## 🔍 Feature Importance

### Top Risk Factors (Increase NPL Probability)

| Rank | Feature | Impact |
|------|---------|--------|
| 1 | High `debt_to_equity` ratio | +++ |
| 2 | Low `credit_bureau_score` | +++ |
| 3 | High `loan_to_revenue_ratio` | ++ |
| 4 | Previous loan defaults | ++ |
| 5 | Short `years_in_operation` | + |

### Protective Factors (Decrease NPL Probability)

| Rank | Feature | Impact |
|------|---------|--------|
| 1 | High `collateral_coverage_pct` | --- |
| 2 | Strong `current_ratio` | --- |
| 3 | High `profit_margin_pct` | -- |
| 4 | Long banking relationship | -- |
| 5 | Higher owner education | - |

---

## 💡 Business Recommendations

### 1. Risk-Based Approval Thresholds

| Risk Score | Action | Description |
|------------|--------|-------------|
| 75-100 | ✅ Auto-Approve | Low risk, expedite processing |
| 50-74 | 🔍 Manual Review | Moderate risk, analyst review |
| 25-49 | ⚠️ Senior Review | High risk, senior approval needed |
| 0-24 | ❌ Auto-Decline | Very high risk, reject application |

### 2. Enhanced Due Diligence Triggers

Flag applications for additional review when:
- Debt-to-equity ratio > 2.0
- Credit bureau score < 500
- Loan-to-revenue ratio > 0.5
- Business operating < 2 years
- Previous loan defaults > 0

### 3. Monitoring & Maintenance

| Activity | Frequency | Purpose |
|----------|-----------|---------|
| Performance monitoring | Monthly | Track accuracy, drift |
| Model recalibration | Quarterly | Update with new data |
| Full retraining | Annually | Incorporate new patterns |
| Fairness audit | Semi-annually | Check for bias across regions/sectors |

---

## 🔮 Future Improvements

### Short-term Enhancements

- [ ] Add cross-validation for more robust evaluation
- [ ] Implement hyperparameter tuning (GridSearchCV)
- [ ] Add more ensemble methods (Random Forest, XGBoost)
- [ ] Create interactive dashboard for model monitoring

### Medium-term Enhancements

- [ ] Incorporate time-series features (payment patterns)
- [ ] Add external data sources (economic indicators)
- [ ] Develop API for real-time scoring
- [ ] Implement model explainability (SHAP values)

### Long-term Vision

- [ ] Deep learning models for complex patterns
- [ ] Automated retraining pipeline
- [ ] A/B testing framework for model updates
- [ ] Integration with core banking systems

---

## 📊 Generated Outputs
### Visualizations (saved to `outputs/` folder)

| File | Description |
|------|-------------|
| `01_numerical_distributions.png` | Histograms of all numerical features |
| `02_categorical_distributions.png` | Bar charts of categorical features |
| `03_npl_rate_by_categories.png` | NPL rate breakdown by category |
| `04_numerical_by_status.png` | Box plots comparing features by loan status |
| `05_correlation_heatmap.png` | Feature correlation matrix |
| `06_model_comparison.png` | Bar chart comparing model metrics |
| `07_roc_curves.png` | ROC curves for all models |
| `08_business_cost_comparison.png` | Business cost by model |
| `09_feature_importance.png` | Top features from Logistic Regression |

### Model Artifacts (saved to `models/` folder)

| File | Contents |
|------|----------|
| `sme_credit_risk_model.pkl` | Trained model, scaler, feature names, encoders, metrics |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Include comments for complex logic
- Write unit tests for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Project Maintainer:** Quayefio  
**Email:** dquayefio813@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/david-quayefio/

---

## 🙏 Acknowledgments

- Ghana banking sector data patterns
- scikit-learn documentation and community
- imbalanced-learn library for SMOTE implementation
- Seaborn for visualization templates

---

*Built with ❤️ for Ghana's financial sector*
