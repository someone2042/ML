"""
Bank Marketing - maximum-effort pipeline
========================================
Feature engineering -> Optuna (LightGBM, XGBoost, CatBoost) -> repeated-CV seed bagging
-> stacking -> threshold chosen on OUT-OF-FOLD predictions -> ONE final evaluation on a
held-out test set that is never used for tuning, model selection or thresholding.

Install:  pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna
Run:      python train_extreme.py                      # full run (hours on CPU)
          python train_extreme.py --trials-scale 0.1   # quick smoke test
          python train_extreme.py --drop-duration      # realistic "before the call" model

Optuna studies are stored in SQLite (results/optuna.db), so you can stop and resume.
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------------------------------- config
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="bank-full.csv")
ap.add_argument("--out", default="results")
ap.add_argument("--drop-duration", action="store_true",
                help="Remove call duration (unknown before the call => leakage for real use)")
ap.add_argument("--no-time-features", action="store_true")
ap.add_argument("--trials-scale", type=float, default=1.0)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024, 1337, 99])
args = ap.parse_args()

SEED = 42
TRIALS = {"lgb": int(400 * args.trials_scale),
          "xgb": int(300 * args.trials_scale),
          "cat": int(80 * args.trials_scale)}
os.makedirs(args.out, exist_ok=True)

CAT_COLS = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "poutcome", "job_edu"]
MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}


# ----------------------------------------------------------------------------- features
def load_and_engineer():
    df = pd.read_csv(args.data, sep=";")
    y = (df.pop("y") == "yes").astype(int).values

    df["month_num"] = df["month"].map(MONTHS)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    if not args.no_time_features:
        # bank-full.csv is stored chronologically (May 2008 -> Nov 2010): a month that
        # goes "backwards" means a new year. Only valid if new data is scored in time order.
        year_idx = (df["month_num"].diff().fillna(0) <= -3).cumsum()
        date = pd.to_datetime(dict(year=2008 + year_idx, month=df["month_num"], day=df["day"]),
                              errors="coerce")
        df["days_since_start"] = (date - date.min()).dt.days
        df["weekday"] = date.dt.dayofweek
        df["row_order"] = np.arange(len(df)) / len(df)

    # pdays == -1 means "never contacted"
    df["pdays_missing"] = (df["pdays"] == -1).astype(int)
    df["pdays"] = df["pdays"].where(df["pdays"] != -1, np.nan)
    df["prev_success"] = (df["poutcome"] == "success").astype(int)

    df["balance_slog"] = np.sign(df["balance"]) * np.log1p(df["balance"].abs())
    df["debt_count"] = (df[["default", "housing", "loan"]] == "yes").sum(axis=1)
    df["edu_ord"] = df["education"].map({"primary": 1, "secondary": 2, "tertiary": 3})
    df["balance_per_age"] = df["balance"] / df["age"]
    df["total_contacts"] = df["campaign"] + df["previous"]
    df["job_edu"] = df["job"] + "_" + df["education"]

    if args.drop_duration:
        df = df.drop(columns=["duration"])
    else:
        df["duration_log"] = np.log1p(df["duration"])
        df["dur_per_call"] = df["duration"] / df["campaign"]

    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df, y


def as_str_frame(X):
    Xs = X.copy()
    for c in CAT_COLS:
        Xs[c] = Xs[c].astype(str)
    return Xs


# ----------------------------------------------------------------------------- models
def build(name, params, seed):
    if name == "lgb":
        return lgb.LGBMClassifier(objective="binary", metric="average_precision",
                                  n_estimators=10000, subsample_freq=1, random_state=seed,
                                  n_jobs=-1, verbose=-1, **params)
    if name == "xgb":
        return xgb.XGBClassifier(tree_method="hist", enable_categorical=True,
                                 eval_metric="aucpr", n_estimators=10000,
                                 early_stopping_rounds=200, random_state=seed,
                                 n_jobs=-1, **params)
    return CatBoostClassifier(iterations=10000, eval_metric="AUC", early_stopping_rounds=200,
                              random_seed=seed, verbose=False, thread_count=-1,
                              cat_features=CAT_COLS, **params)


def fit(name, m, Xtr, ytr, Xva, yva):
    if name == "lgb":
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(200, verbose=False)])
    elif name == "xgb":
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
    return m


def space(name, t):
    if name == "lgb":
        return dict(
            learning_rate=t.suggest_float("learning_rate", 0.005, 0.08, log=True),
            num_leaves=t.suggest_int("num_leaves", 8, 256, log=True),
            max_depth=t.suggest_int("max_depth", 3, 12),
            min_child_samples=t.suggest_int("min_child_samples", 5, 200, log=True),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=t.suggest_float("reg_alpha", 1e-8, 10, log=True),
            reg_lambda=t.suggest_float("reg_lambda", 1e-8, 10, log=True),
            min_split_gain=t.suggest_float("min_split_gain", 0.0, 1.0),
            scale_pos_weight=t.suggest_float("scale_pos_weight", 1.0, 8.0),
            cat_smooth=t.suggest_float("cat_smooth", 1, 100),
            cat_l2=t.suggest_float("cat_l2", 1, 100),
            max_bin=t.suggest_int("max_bin", 64, 512),
        )
    if name == "xgb":
        return dict(
            learning_rate=t.suggest_float("learning_rate", 0.005, 0.1, log=True),
            max_depth=t.suggest_int("max_depth", 3, 10),
            min_child_weight=t.suggest_float("min_child_weight", 1, 30, log=True),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 1.0),
            gamma=t.suggest_float("gamma", 1e-8, 5, log=True),
            reg_alpha=t.suggest_float("reg_alpha", 1e-8, 10, log=True),
            reg_lambda=t.suggest_float("reg_lambda", 1e-8, 10, log=True),
            scale_pos_weight=t.suggest_float("scale_pos_weight", 1.0, 8.0),
            max_bin=t.suggest_int("max_bin", 64, 512),
        )
    return dict(
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        depth=t.suggest_int("depth", 4, 10),
        l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 30, log=True),
        random_strength=t.suggest_float("random_strength", 0.1, 10, log=True),
        bagging_temperature=t.suggest_float("bagging_temperature", 0, 5),
        border_count=t.suggest_int("border_count", 32, 255),
        one_hot_max_size=t.suggest_int("one_hot_max_size", 2, 10),
        scale_pos_weight=t.suggest_float("scale_pos_weight", 1.0, 8.0),
    )


# ----------------------------------------------------------------------------- main
def main():
    df, y = load_and_engineer()
    print(f"Features: {df.shape[1]}  Rows: {len(df)}  Positive rate: {y.mean():.3%}")

    X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_te = X_tr.reset_index(drop=True), X_te.reset_index(drop=True)
    data = {"lgb": (X_tr, X_te), "xgb": (X_tr, X_te),
            "cat": (as_str_frame(X_tr), as_str_frame(X_te))}

    # ---------------- 1) Optuna, per model (CV seed differs from the final CV seeds)
    best = {}
    for name in ["lgb", "xgb", "cat"]:
        Xtr_m, _ = data[name]

        def objective(trial, name=name, Xtr_m=Xtr_m):
            params = space(name, trial)
            skf = StratifiedKFold(args.folds, shuffle=True, random_state=123)
            scores = []
            for i, (a, b) in enumerate(skf.split(Xtr_m, y_tr)):
                m = fit(name, build(name, params, SEED),
                        Xtr_m.iloc[a], y_tr[a], Xtr_m.iloc[b], y_tr[b])
                scores.append(average_precision_score(y_tr[b], m.predict_proba(Xtr_m.iloc[b])[:, 1]))
                trial.report(float(np.mean(scores)), i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize", study_name=f"{name}_ap",
            storage=f"sqlite:///{args.out}/optuna.db", load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True, n_startup_trials=25),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=25, n_warmup_steps=1))
        todo = max(0, TRIALS[name] - len(study.trials))
        print(f"[{name}] running {todo} trials ...")
        study.optimize(objective, n_trials=todo, show_progress_bar=True)
        best[name] = study.best_params
        print(f"[{name}] best CV PR-AUC = {study.best_value:.4f}")

    json.dump(best, open(f"{args.out}/best_params.json", "w"), indent=2)

    # ---------------- 2) Final repeated CV: OOF preds (for stacking/threshold) + test preds
    oof, test = {}, {}
    for name in ["lgb", "xgb", "cat"]:
        Xtr_m, Xte_m = data[name]
        oof[name], test[name] = np.zeros(len(y_tr)), np.zeros(len(y_te))
        for seed in args.seeds:
            skf = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
            for a, b in skf.split(Xtr_m, y_tr):
                m = fit(name, build(name, best[name], seed),
                        Xtr_m.iloc[a], y_tr[a], Xtr_m.iloc[b], y_tr[b])
                oof[name][b] += m.predict_proba(Xtr_m.iloc[b])[:, 1] / len(args.seeds)
                test[name] += m.predict_proba(Xte_m)[:, 1] / (args.folds * len(args.seeds))
        print(f"[{name}] OOF PR-AUC={average_precision_score(y_tr, oof[name]):.4f} "
              f"ROC-AUC={roc_auc_score(y_tr, oof[name]):.4f}")

    # ---------------- 3) Candidates: single models, mean blend, logistic stack
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    names = ["lgb", "xgb", "cat"]
    O = np.column_stack([logit(oof[n]) for n in names])
    T = np.column_stack([logit(test[n]) for n in names])

    stacker = LogisticRegression(C=1.0, max_iter=1000)
    stack_oof = cross_val_predict(stacker, O, y_tr,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=999),
                                  method="predict_proba")[:, 1]
    stacker.fit(O, y_tr)
    cands = {n: (oof[n], test[n]) for n in names}
    cands["mean_blend"] = (np.mean([oof[n] for n in names], 0), np.mean([test[n] for n in names], 0))
    cands["stack"] = (stack_oof, stacker.predict_proba(T)[:, 1])

    scores = {k: average_precision_score(y_tr, v[0]) for k, v in cands.items()}
    print("OOF PR-AUC by candidate:", {k: round(v, 4) for k, v in scores.items()})
    winner = max(scores, key=scores.get)          # chosen on OOF only, never on test
    o, t = cands[winner]
    print("Selected:", winner)

    # ---------------- 4) Threshold from OOF (max F1); swap in your own cost function if needed
    p, r, th = precision_recall_curve(y_tr, o)
    f1s = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    thr = float(th[np.argmax(f1s)])
    print(f"Threshold (OOF-optimal F1): {thr:.4f}  OOF F1={f1s.max():.4f}")

    # ---------------- 5) Single, final evaluation on the untouched test set
    pred = (t >= thr).astype(int)
    res = dict(model=winner, threshold=thr,
               roc_auc=roc_auc_score(y_te, t), pr_auc=average_precision_score(y_te, t),
               accuracy=accuracy_score(y_te, pred), precision=precision_score(y_te, pred),
               recall=recall_score(y_te, pred), f1=f1_score(y_te, pred),
               confusion_matrix=confusion_matrix(y_te, pred).tolist(),
               drop_duration=args.drop_duration, time_features=not args.no_time_features)
    print(json.dumps(res, indent=2))
    json.dump(res, open(f"{args.out}/final_test_metrics.json", "w"), indent=2)
    np.save(f"{args.out}/test_probs.npy", t)
    np.save(f"{args.out}/oof_probs.npy", o)


if __name__ == "__main__":
    main()