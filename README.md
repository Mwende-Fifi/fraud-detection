# Fraud Detection — Credit Card Transactions

Binary classification on the Kaggle [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset: 284,807 transactions, 31 columns, **0.173% fraud**, zero nulls.

**Goal:** catch fraud without flagging everything.

## Approach

- **Split:** stratified 70/15/15, `random_state=42`. Test set sealed until final evaluation.
- **Models:** Logistic Regression baseline (StandardScaler fit on train only) vs XGBoost (`n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=578.55`).
- **Threshold:** swept 0.00→1.00 on the validation set, chose **0.75** (recall-weighted with a false-alarm cap).

## Results

| Model | Set | AUC | Threshold | Precision | Recall | FP | FN |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Val | 0.9571 | 0.50 | 0.786 | 0.595 | 12 | 30 |
| XGBoost | Val | 0.9765 | 0.50 | 0.571 | 0.811 | 45 | 14 |
| **XGBoost** | **Test** | **0.9571** | **0.75** | **0.819** | **0.797** | **13** | **15** |

## Key finding

Accuracy is a trap here: the model scores 99.93% on test, versus 99.83% for an all-legit classifier — a 0.1pp gap that hides the real story (13 false alarms, 15 missed frauds). On the sealed test set XGBoost's AUC (0.9571) matched logistic regression's validation AUC, suggesting the model's validation edge was partly small-sample noise. **Threshold choice drove the operating point more than model choice.**

## Threshold justification

Swept the XGBoost fraud probabilities from 0.00 to 1.00 and compared three operating points. At the default 0.5 the model caught 60 of 74 frauds but raised 45 false alarms — too heavy a workload for a 0.173% fraud base rate. The max-F1 point (t=0.96) is the cleanest statistically (precision 0.915, only 5 false alarms) but misses 20 frauds, giving up real catches to look tidy. The ≥90%-recall point (t=0.01) catches almost everything (6 missed) but flags 2,148 legitimate transactions — an unusable alert queue. Chose **t=0.75**: false alarms held to 20 (down from 45 at default), precision 0.737, 18 frauds missed — a recall-weighted compromise with a hard cap on investigation load. Catches 2 more frauds than max-F1 for 15 more alerts, a trade worth taking because a missed fraud is a direct loss while a false alarm is an investigation cost. The honest caveat: the right point depends on the actual dollar cost of a missed fraud versus a false alarm, which isn't quantified here — that's Project 2's job. If false alarms carried reputational or customer-friction cost, t=0.96 would win instead.

## Conclusion

**What worked:** Stratified splitting plus `scale_pos_weight` let XGBoost catch ~80% of frauds at a threshold that kept the alert queue to 13 false alarms on the sealed test set. **What didn't:** The model's validation AUC (0.9765) did not hold on test (0.9571) — the same score logistic regression reached on validation — showing that with only 74 frauds in validation, the XGBoost edge was partly noise. **What I learned:** In extreme-imbalance problems the honest number is the one from data you never touched, and the real decision isn't the model — it's where to set the threshold, which depends on a cost ratio not yet quantified.

## Repo layout

fraud-detection/
├── 01_fraud.ipynb # full analysis: EDA, splits, models, threshold sweep, test eval
├── README.md
└── .gitignore # excludes raw data, splits, secrets

Data and splits are gitignored — download the dataset from Kaggle and run the notebook to regenerate them.

