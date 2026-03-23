#!/usr/bin/env python3
"""PhD-level exploratory analysis for diabetes_graph_data.csv.

Steps covered (modular so you can cherry-pick):
- data audit (missingness, duplicates, logical checks)
- distribution summaries with robust stats and outlier flags
- target balance and binary prevalence
- risk tables with risk difference / ratio / odds ratio + CIs
- ordinal trend tests (via logistic slope p-value)
- correlation heatmap (Spearman) and partial dependence style plots
- baseline logistic model + calibration + decision curve
- evaluation of existing rule_pred

Run as a script or open in VS Code/Jupyter using # %% cells.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import calibration, linear_model, metrics, model_selection, preprocessing

sns.set_theme(style="whitegrid")

DATA_PATH = Path("Health Care/diabetes_graph_data.csv")
FIG_DIR = Path("Health Care/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Diabetes_binary"
BINARY_COLS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "DiffWalk",
    "rule_pred",
]
CONT_COLS = ["BMI", "PhysHlth"]
ORDINAL_COLS = ["GenHlth", "Age", "Education", "Income", "BMI_cat_code"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    # Cast integer-like columns to Int64 for clean counts
    for col in BINARY_COLS + ORDINAL_COLS + [TARGET]:
        if col in df.columns:
            df[col] = df[col].round().astype("Int64")
    return df


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().rename("n_missing")
    miss_pct = (miss / len(df) * 100).rename("pct")
    out = pd.concat([miss, miss_pct], axis=1)
    out.to_csv(FIG_DIR / "missing_summary.csv")
    return out


def duplicate_report(df: pd.DataFrame) -> int:
    dup_count = df.duplicated().sum()
    (FIG_DIR / "duplicate_report.txt").write_text(f"Duplicate rows: {dup_count}\n")
    return dup_count


def logical_checks(df: pd.DataFrame) -> pd.DataFrame:
    issues = {}
    if {"HighChol", "CholCheck"}.issubset(df):
        mask = (df["CholCheck"] == 0) & (df["HighChol"] == 1)
        issues["HighChol_without_screen"] = int(mask.sum())
    if {"HeartDiseaseorAttack", "HighBP"}.issubset(df):
        # high cardiovascular disease with no high BP can still happen, so we just count
        mask = (df["HeartDiseaseorAttack"] == 1) & (df["HighBP"] == 0)
        issues["Cardio_no_HighBP"] = int(mask.sum())
    if issues:
        pd.Series(issues).to_csv(FIG_DIR / "logical_checks.csv")
    return pd.DataFrame.from_dict(issues, orient="index", columns=["count"])


def robust_describe(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        x = df[col].dropna()
        if x.empty:
            continue
        median = x.median()
        mad = stats.median_abs_deviation(x, scale="normal")
        boot = bootstrap_ci(x, np.mean)
        rows.append(
            {
                "feature": col,
                "n": len(x),
                "mean": x.mean(),
                "mean_ci_low": boot[0],
                "mean_ci_high": boot[1],
                "median": median,
                "mad": mad,
                "p05": x.quantile(0.05),
                "p95": x.quantile(0.95),
                "skew": stats.skew(x),
                "kurtosis": stats.kurtosis(x),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(FIG_DIR / "robust_summary.csv", index=False)
    return out


def bootstrap_ci(x: pd.Series, stat_fn, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    x = np.asarray(x.dropna())
    rng = np.random.default_rng(42)
    stats_boot = []
    for _ in range(n_boot):
        resample = rng.choice(x, size=len(x), replace=True)
        stats_boot.append(stat_fn(resample))
    low, high = np.percentile(stats_boot, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(low), float(high)


def outlier_flags(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        x = df[col].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        rows.append({"feature": col, "n_outliers": int(outlier_mask.sum()), "pct": outlier_mask.mean() * 100})
    out = pd.DataFrame(rows)
    out.to_csv(FIG_DIR / "outlier_report.csv", index=False)
    return out


def prevalence_plot(df: pd.DataFrame, cols: Iterable[str]) -> None:
    prev = df[list(cols)].mean().sort_values()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=prev.values, y=prev.index, orient="h", palette="crest")
    plt.xlabel("Prevalence")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "binary_prevalence.png", dpi=200)
    plt.close()


def target_balance(df: pd.DataFrame) -> pd.Series:
    counts = df[TARGET].value_counts(dropna=False)
    counts.to_csv(FIG_DIR / "target_balance.csv")
    plt.figure(figsize=(4, 3))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="flare")
    plt.ylabel("Count")
    plt.xlabel("Diabetes_binary")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_balance.png", dpi=200)
    plt.close()
    return counts


@dataclass
class RiskMetrics:
    risk_exposed: float
    risk_unexposed: float
    risk_diff: float
    risk_diff_ci: Tuple[float, float]
    risk_ratio: float
    risk_ratio_ci: Tuple[float, float]
    odds_ratio: float
    odds_ratio_ci: Tuple[float, float]
    n_exposed: int
    n_unexposed: int


def risk_table(df: pd.DataFrame, exposures: Iterable[str]) -> pd.DataFrame:
    rows: Dict[str, RiskMetrics] = {}
    for col in exposures:
        if col not in df:
            continue
        rows[col] = compute_risk_metrics(df[col], df[TARGET])
    out = pd.DataFrame.from_dict({k: vars(v) for k, v in rows.items()}, orient="index")
    out.to_csv(FIG_DIR / "risk_table.csv")
    return out


def compute_risk_metrics(exposure: pd.Series, outcome: pd.Series) -> RiskMetrics:
    data = pd.DataFrame({"e": exposure, "y": outcome}).dropna()
    a = ((data["e"] == 1) & (data["y"] == 1)).sum()
    b = ((data["e"] == 1) & (data["y"] == 0)).sum()
    c = ((data["e"] == 0) & (data["y"] == 1)).sum()
    d = ((data["e"] == 0) & (data["y"] == 0)).sum()
    n1, n0 = a + b, c + d
    # Add 0.5 to avoid zero-division (Haldane-Anscombe correction)
    a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    risk1, risk0 = a / n1 if n1 else np.nan, c / n0 if n0 else np.nan
    rd = risk1 - risk0
    se_rd = np.sqrt(risk1 * (1 - risk1) / n1 + risk0 * (1 - risk0) / n0)
    rd_ci = (rd - 1.96 * se_rd, rd + 1.96 * se_rd)
    rr = (a2 / n1) / (c2 / n0)
    se_log_rr = np.sqrt((1 / a2) - (1 / n1) + (1 / c2) - (1 / n0))
    rr_ci = (np.exp(np.log(rr) - 1.96 * se_log_rr), np.exp(np.log(rr) + 1.96 * se_log_rr))
    or_ = (a2 * d2) / (b2 * c2)
    se_log_or = np.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2)
    or_ci = (np.exp(np.log(or_) - 1.96 * se_log_or), np.exp(np.log(or_) + 1.96 * se_log_or))
    return RiskMetrics(risk1, risk0, rd, rd_ci, rr, rr_ci, or_, or_ci, n1, n0)


def ordinal_trend_tests(df: pd.DataFrame, ordinals: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in ordinals:
        if col not in df:
            continue
        x = df[[col, TARGET]].dropna()
        if x[col].nunique() < 3:
            continue
        x = x.assign(const=1)
        # Logistic regression slope p-value as trend test
        model = linear_model.LogisticRegression(penalty="none", solver="lbfgs", max_iter=200)
        model.fit(x[[col]], x[TARGET])
        # Wald z for slope
        # Approximated using statsmodels-like formula
        probs = model.predict_proba(x[[col]])[:, 1]
        X = np.column_stack([np.ones(len(x)), x[col]])
        W = np.diag(probs * (1 - probs))
        cov = np.linalg.inv(X.T @ W @ X)
        se = np.sqrt(np.diag(cov))[1]
        z = model.coef_[0][0] / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        rows.append({"feature": col, "coef": model.coef_[0][0], "z": z, "p_value": p})
    out = pd.DataFrame(rows)
    out.to_csv(FIG_DIR / "ordinal_trend_tests.csv", index=False)
    return out


def correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[BINARY_COLS + CONT_COLS + ORDINAL_COLS].corr(method="spearman")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "spearman_corr.png", dpi=200)
    plt.close()


def target_vs_continuous(df: pd.DataFrame, cont_cols: Iterable[str]) -> None:
    for col in cont_cols:
        if col not in df:
            continue
        plt.figure(figsize=(5, 4))
        sns.boxplot(data=df, x=TARGET, y=col, palette="Set2")
        sns.stripplot(data=df.sample(min(2000, len(df)), random_state=42), x=TARGET, y=col, color="0.2", size=1, alpha=0.4)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{col}_by_target.png", dpi=200)
        plt.close()


def interaction_heatmap(df: pd.DataFrame, col_x: str, col_y: str) -> None:
    if not {col_x, col_y, TARGET}.issubset(df.columns):
        return
    pivot = pd.crosstab(df[col_y], df[col_x], df[TARGET], aggfunc="mean")
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="magma", cbar_kws={"label": "P(Diabetes=1)"})
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"interaction_{col_x}_{col_y}.png", dpi=200)
    plt.close()


def rule_pred_evaluation(df: pd.DataFrame) -> Dict[str, float]:
    if "rule_pred" not in df:
        return {}
    y_true = df[TARGET]
    y_pred = df["rule_pred"]
    cm = metrics.confusion_matrix(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    np.savetxt(FIG_DIR / "rule_pred_cm.csv", cm, delimiter=",", fmt="%d")
    pd.DataFrame(report).to_csv(FIG_DIR / "rule_pred_report.csv")
    return {"accuracy": report["accuracy"], "sensitivity": report["1"]["recall"], "specificity": report["0"]["recall"]}


def baseline_logistic(df: pd.DataFrame) -> Dict[str, float]:
    features = BINARY_COLS[:-1] + ORDINAL_COLS + CONT_COLS  # exclude rule_pred from model
    X = df[features].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = preprocessing.StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = linear_model.LogisticRegression(max_iter=300, penalty="l2", n_jobs=-1)
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)[:, 1]
    preds = (proba >= 0.5).astype(int)
    auc = metrics.roc_auc_score(y_test, proba)
    pr_auc = metrics.average_precision_score(y_test, proba)
    brier = metrics.brier_score_loss(y_test, proba)
    calib = calibration.calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(4, 4))
    plt.plot(calib[1], calib[0], marker="o", label="model")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="ideal")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "baseline_calibration.png", dpi=200)
    plt.close()
    metrics_dict = {"roc_auc": auc, "pr_auc": pr_auc, "brier": brier}
    pd.Series(metrics_dict).to_csv(FIG_DIR / "baseline_metrics.csv")
    return metrics_dict


def decision_curve(y_true: pd.Series, proba: np.ndarray, label: str) -> pd.DataFrame:
    thresholds = np.linspace(0.01, 0.99, 50)
    nb = []
    prevalence = y_true.mean()
    for t in thresholds:
        tp = ((proba >= t) & (y_true == 1)).mean()
        fp = ((proba >= t) & (y_true == 0)).mean()
        net_benefit = tp - fp * (t / (1 - t))
        nb.append({"threshold": t, "net_benefit": net_benefit})
    out = pd.DataFrame(nb)
    out["label"] = label
    return out


def run_decision_curve(df: pd.DataFrame) -> None:
    features = BINARY_COLS[:-1] + ORDINAL_COLS + CONT_COLS
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        df[features], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    scaler = preprocessing.StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = linear_model.LogisticRegression(max_iter=300, penalty="l2", n_jobs=-1)
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)[:, 1]
    nb_model = decision_curve(y_test, proba, label="logit")
    nb_rule = decision_curve(df[TARGET], df["rule_pred"], label="rule_pred") if "rule_pred" in df else pd.DataFrame()
    nb = pd.concat([nb_model, nb_rule], ignore_index=True)
    plt.figure(figsize=(6, 4))
    for label, g in nb.groupby("label"):
        plt.plot(g["threshold"], g["net_benefit"], label=label)
    plt.axhline(0, color="grey", linestyle="--", label="treat none")
    plt.axhline(df[TARGET].mean(), color="black", linestyle=":", label="treat all")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "decision_curve.png", dpi=200)
    plt.close()


def main():
    df = load_data()
    print("Loaded", df.shape)

    summarize_missing(df)
    duplicate_report(df)
    logical_checks(df)

    robust_describe(df, CONT_COLS + ORDINAL_COLS)
    outlier_flags(df, CONT_COLS)

    target_balance(df)
    prevalence_plot(df, BINARY_COLS)
    risk_table(df, BINARY_COLS)
    ordinal_trend_tests(df, ORDINAL_COLS)
    correlation_heatmap(df)
    target_vs_continuous(df, CONT_COLS)
    interaction_heatmap(df, "HighBP", "BMI_cat_code")

    rule_pred_evaluation(df)
    baseline_logistic(df)
    run_decision_curve(df)

    print("Artifacts saved to", FIG_DIR.resolve())


if __name__ == "__main__":
    main()

