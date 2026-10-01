# Titanic: EDA + a Simple Survival Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laudwally/titanic-eda-ml/blob/main/notebook.ipynb)

A self-contained data analysis + machine learning notebook: load the classic Titanic passenger
dataset, explore it, engineer a couple of features, train and compare two classifiers, and
evaluate the better one — all runnable top-to-bottom with no API keys and no local files (the
dataset loads from a public URL).

## What's in it

1. **Load** — pulls `titanic.csv` directly from a public GitHub-hosted copy of the dataset
2. **EDA** — missing-data breakdown, survival rate by sex / class / age, a combined class×sex view
3. **Feature engineering** — `FamilySize` and `IsAlone` derived from sibling/spouse and
   parent/child counts
4. **Modeling** — Logistic Regression vs. Random Forest, both through the same
   impute → scale/encode preprocessing pipeline
5. **Evaluation** — accuracy, a full classification report, and a confusion matrix for the
   better-performing model

Headline result in the committed run: Random Forest, 81.0% test accuracy.

## Running it

**In Colab** (easiest — click the badge above, then Runtime → Run all):

The notebook has no setup cell because Colab already has pandas/numpy/matplotlib/seaborn/
scikit-learn preinstalled.

**Locally:**

```bash
python -m venv .venv
.venv/Scripts/activate      # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Saving your own changes back to GitHub from Colab

Colab's **File → Save a copy in GitHub** (or **File → Save**, if you opened it from GitHub and
have write access) commits directly to this repo from your Google account — no local git setup
needed.
