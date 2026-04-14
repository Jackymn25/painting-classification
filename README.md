# Artwork Classification

A machine learning project for classifying paintings into three target classes using structured features, multi-label categorical features, and open-text responses.
final model stats: **traning accuracy: 100% validation accuracy(5-fold): 92% test accuracy: 91%**

## Overview

This project builds a multi-class artwork classifier on a mixed-type dataset containing numeric features, multi-label categorical features, and open-text features.

The pipeline combines traditional feature engineering with probabilistic text modeling. Instead of feeding raw text directly into the final classifier, each text column is first processed by a Bernoulli Naive Bayes sub-model, and the resulting probability features are merged with the structured inputs.

## Dataset

The dataset contains:

- 1,686 samples
- 14 features

Feature types include:

- ordered numeric ratings
- count-based variables
- multi-select categorical variables
- three open-text columns

## Data Processing

The project includes the following preprocessing steps:

- removed rows with severe missingness
- checked that class balance remained stable after cleaning
- kept ordered numeric variables as numeric instead of one-hot encoding
- capped upper-tail outliers for count-based features
- cleaned inconsistent numeric text formats in the willingness-to-pay field
- encoded multi-label categorical variables as multi-hot vectors
- cleaned text using lowercase conversion, punctuation removal, stopword removal, and light normalization
- built top-K vocabularies from training data only
- converted text columns into Bernoulli Naive Bayes probability features

## Models Explored

The following model families were tested on the same feature representation:

- Logistic Regression + Naive Bayes text features
- Random Forest + Naive Bayes text features
- K-Nearest Neighbors
- Multi-Layer Perceptron

All model families were compared under the same validation setup.

## Final Model

The final submitted model is:

- Bernoulli Naive Bayes for the three text columns
- Multinomial Logistic Regression as the outer classifier

### Final hyperparameters

- `food_k = 38`
- `feeling_k = 44`
- `soundtrack_k = 62`
- `nb_alpha = 0.5`
- `C = 0.5`

## Performance

- Best mean validation accuracy: **0.9174**
- Validation standard deviation: **0.0072**
- Conservative estimated test accuracy: **0.90**

Although one held-out test run produced a higher score, the project reports `0.90` as a more cautious estimate.

## How It Works

1. Clean and preprocess numeric, categorical, and text features.
2. Build top-K vocabularies for each text column using training data.
3. Fit Bernoulli Naive Bayes models on the text columns.
4. Convert text into class-probability-based features.
5. Concatenate numeric features, multi-hot categorical features, and NB-derived text features.
6. Standardize the full feature matrix.
7. Fit multinomial logistic regression.
8. Use the trained pipeline in `pred.py` to make predictions.

## How to Run

Run the prediction script:

```bash
python xxx.py
```

If the script exposes a function like `predict_all(filename)`, use it like this:

```python
from xxx import predict_all

preds = predict_all("test.csv")
print(preds[:10])
```

## Notes

- The text features are not used as raw text in the final classifier.
- All models were compared on a common representation for fairness.
- Hyperparameters were tuned with 5-fold cross-validation on the training/validation split.
- The test set was kept separate during model selection.

## Main Idea

The main design choice in this project is to treat text as a structured probabilistic signal rather than an unprocessed raw input. This keeps the model simpler, more interpretable, and easier to compare across classifier families.

## Authors

- Haozhe Huo
- Jingcheng Liang
- Xun Tang

