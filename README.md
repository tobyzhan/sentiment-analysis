# HCAHPS Patient Survey Sentiment Analysis 🏥

A machine learning pipeline that performs sentiment analysis on California HCAHPS (Hospital Consumer Assessment of Healthcare Providers and Systems) patient survey data. Used for 2024 SDSU Big Data Hackathon

---

## What It Does

- Loads and preprocesses California HCAHPS survey data
- Uses **TextBlob** to compute sentiment polarity scores on survey questions
- Categorizes responses as **Positive**, **Neutral**, or **Negative** based on answer percentages
- Trains a **Logistic Regression** classifier combining TF-IDF text features with numeric survey features
- Visualizes results with confusion matrices, scatter plots, and sentiment distribution bar charts

---

## Dataset

**File:** `Patient Survey HCAHPS California.csv`

Key columns used:
| Column | Description |
|---|---|
| `HCAHPS Question` | The survey question text |
| `HCAHPS Answer Percent` | Percentage of respondents giving that answer |
| `Number of Completed Surveys` | Total survey count for that hospital |

---

## Pipeline

```
Raw CSV
  ↓
TextBlob sentiment polarity scoring
  ↓
Rule-based sentiment labeling (based on Answer Percent)
  ↓
TF-IDF vectorization of question text
  ↓
Combine with numeric features (sparse matrix)
  ↓
Logistic Regression classifier
  ↓
Evaluation + Visualizations
```

---

## Sentiment Labeling Rules

| Answer Percent | Label |
|---|---|
| ≥ 70% | Positive |
| 20% – 69% | Neutral |
| < 20% | Negative |

---

## Model

- **Algorithm:** Logistic Regression (`max_iter=10000`)
- **Text features:** TF-IDF (`max_features=5000`)
- **Numeric features:** Number of Completed Surveys
- **Combination:** Sparse matrix horizontal stack (`scipy.sparse.hstack`)
- **Train/test split:** 80/20

---

## Visualizations

1. **Confusion Matrix** — Heatmap of predicted vs true sentiment labels
2. **Scatter Plot** — Sentiment Score vs Answer Percentage, point size scaled by number of respondents
3. **Bar Chart** — Sentiment distribution with sample survey questions annotated above each bar

---

## Requirements

```bash
pip install pandas numpy scikit-learn scipy textblob matplotlib seaborn
python -m textblob.download_corpora
```

---

## Usage

1. Place `Patient Survey HCAHPS California.csv` in the same directory as the notebook
2. Open `sentiment_analysis.ipynb` in Jupyter or VS Code
3. Run all cells top to bottom

---

## Results

The model outputs:
- Accuracy score
- Full classification report (precision, recall, F1 per class)
-
