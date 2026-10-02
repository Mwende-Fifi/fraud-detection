
# Fraud Detection — Credit Card Transactions

Binary classification on the Kaggle [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset: 284,807 transactions, 31 columns, 0.173% fraud, zero nulls.

Goal: catch fraud without flagging everything.

## Approach

- Split: stratified 70/15/15, `random_state=42`. Test set sealed until final evaluation.
- Models: logistic regression baseline (StandardScaler fit on train only) vs XGBoost (`n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=578.55`).
- Threshold: swept 0.00 to 1.00 on the validation set, chose 0.75.

## Results

| Model | Set | AUC | Threshold | Precision | Recall | FP | FN |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Val | 0.9571 | 0.50 | 0.786 | 0.595 | 12 | 30 |
| XGBoost | Val | 0.9765 | 0.50 | 0.571 | 0.811 | 45 | 14 |
| XGBoost | Test | 0.9571 | 0.75 | 0.819 | 0.797 | 13 | 15 |

## Key finding

Accuracy is a trap here. The model scores 99.93% on test against 99.83% for an all-legit classifier, a 0.1pp gap that hides the real story: 13 false alarms and 15 missed frauds. On the sealed test set, XGBoost's AUC (0.9571) matched logistic regression's validation AUC, which suggests the model's validation edge was partly small-sample noise. Threshold choice moved the operating point more than model choice did.

## Threshold choice

I swept the XGBoost probabilities from 0.00 to 1.00 and compared three operating points. At the default 0.5 the model caught 60 of 74 frauds but raised 45 false alarms, too heavy for a 0.173% fraud base rate. Max F1 (t=0.96) is statistically cleanest at precision 0.915 and only 5 false alarms, but misses 20 frauds. The 90%-recall point (t=0.01) catches almost everything, 6 missed, but flags 2,148 legitimate transactions, which is an unusable queue. I chose 0.75: false alarms held to 20, precision 0.737, 18 frauds missed. It catches 2 more frauds than max F1 for 15 more alerts. I took that trade because a missed fraud is a direct loss while a false alarm is an investigation cost. The caveat: the right point depends on the dollar cost of each, which I have not quantified. That is Project 2. If false alarms carried reputational cost, t=0.96 would win instead.

## Conclusion

What worked: stratified splitting plus `scale_pos_weight` let XGBoost catch about 80% of frauds at a threshold that kept the test-set alert queue to 13 false alarms. What didn't: the validation AUC (0.9765) did not hold on test (0.9571), the same score logistic regression reached on validation, so the XGBoost edge was partly noise from only 74 validation frauds. What I learned: in extreme imbalance the honest number comes from data you never touched, and the real decision is not the model but where to set the threshold, which depends on a cost ratio I have not quantified yet.

## Repo layout

```
fraud-detection/
├── 01_fraud.ipynb    # EDA, splits, models, threshold sweep, test evaluation
├── README.md
└── .gitignore        # excludes raw data, splits, secrets
```

Data and splits are gitignored. Download the dataset from Kaggle and run the notebook to regenerate them.
