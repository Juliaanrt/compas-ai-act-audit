"""
02 - Predictive performance across model x feature-set combinations
(Section 4.1, Table 2).

Logistic regression, random forest and XGBoost with library defaults on
the three nested feature sets, evaluated on the seed-42 80/20 split.

Outputs: outputs/table2_performance.csv
"""

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import config as C
from common import MODEL_LABELS, encode, load_compas, make_model, save, split


def main():
    df = load_compas()
    y = df[C.TARGET].astype(int)
    idx_tr, idx_te = split(df, C.SEED_DESCRIPTIVE)
    rows = []
    for fs_name, cols in C.FEATURE_SETS.items():
        X = encode(df, cols)
        for m in ["lr", "rf", "xgb"]:
            model = make_model(m, C.SEED_DESCRIPTIVE).fit(X.loc[idx_tr], y.loc[idx_tr])
            rows.append({"Model": MODEL_LABELS[m], "Feature set": fs_name,
                         "Accuracy": round(accuracy_score(y.loc[idx_te], model.predict(X.loc[idx_te])), 3),
                         "AUC": round(roc_auc_score(y.loc[idx_te], model.predict_proba(X.loc[idx_te])[:, 1]), 3)})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    save(tab, "table2_performance.csv")


if __name__ == "__main__":
    main()
