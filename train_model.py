"""Re-trains the model and saves bill_model.joblib (same code as the notebook). Run: python train_model.py"""
import matplotlib; matplotlib.use("Agg")
display = print
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from bill_utils import (add_lag_features, FeatureEngineer, TARGET, CAT_ONEHOT, CAT_ORDINAL,
                        HOUSE_ORDER, NUM_COLS, INPUT_COLS)
pd.set_option("display.max_columns", 30)
RANDOM_STATE = 42

df = pd.read_csv("electricity_bills_simple.csv")
print(df.shape)
df.head()

df.info()

print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0], "\n")
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate (customer, month):", df.duplicated(["customer_id", "month"]).sum())
print("Months per customer:", df.groupby("customer_id").month.nunique().describe()[["min", "max"]].to_dict())
print("Month range:", df.month.min(), "->", df.month.max())

# business-rule range checks
checks = {
    "bill_amount > 0": (df.bill_amount > 0).all(),
    "next_month_bill > 0": (df.next_month_bill > 0).all(),
    "family_size 1-10": df.family_size.dropna().between(1, 10).all(),
    "num_acs 0-6": df.num_acs.dropna().between(0, 6).all(),
    "avg_temp 10-45 C": df.avg_temp_c.dropna().between(10, 45).all(),
    "humidity 0-100 %": df.humidity_pct.dropna().between(0, 100).all(),
}
print("\nRange checks:", checks)

for c in ["area", "connection_type", "house_type", "has_solar", "payment_mode"]:
    print(c, "->", df[c].value_counts(dropna=False).to_dict())

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
df[TARGET].plot.hist(bins=40, ax=ax[0], title="Distribution of next month bill (₹)")
df.groupby("month")[TARGET].mean().plot(ax=ax[1], marker="o", title="Average next-month bill by month")
ax[1].tick_params(axis="x", rotation=60)
df.boxplot(column="bill_amount", by="connection_type", ax=ax[2]); ax[2].set_title("Bill by connection type")
plt.suptitle(""); plt.tight_layout(); plt.show()

print(df.groupby(["connection_type", "has_solar"]).bill_amount.median().round(0))
print(df.groupby("house_type").bill_amount.median().round(0))
print(df.groupby("payment_mode").bill_amount.median().round(0))

num = df.select_dtypes("number")
corr = num.corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im); plt.title("Correlation matrix"); plt.tight_layout(); plt.show()

df = add_lag_features(df)
df[["customer_id", "month", "bill_amount", "prev_bill", "rolling_3m_bill", "bill_change", TARGET]].head(6)

train = df[df.month <= "2026-03"].sort_values(["month", "customer_id"])
val   = df[(df.month >= "2026-04") & (df.month <= "2026-06")]
test  = df[df.month >= "2026-07"]
X_train, y_train = train[INPUT_COLS], train[TARGET]
X_val,   y_val   = val[INPUT_COLS],   val[TARGET]
X_test,  y_test  = test[INPUT_COLS],  test[TARGET]
print(f"Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")

print("Missing in TRAIN before imputation:\n", X_train.isna().sum()[X_train.isna().sum() > 0])
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer.fit(X_train[["family_size", "num_acs", "avg_temp_c", "humidity_pct", "units_consumed"]])
print("\nMedians learned from train:", dict(zip(num_imputer.feature_names_in_, num_imputer.statistics_)))

onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ordinal = OrdinalEncoder(categories=HOUSE_ORDER, handle_unknown="use_encoded_value", unknown_value=-1)
demo = onehot.fit(X_train[CAT_ONEHOT].fillna("missing"))
print("One-hot columns:", list(onehot.get_feature_names_out()))

print(X_train[["avg_temp_c", "units_consumed", "bill_amount"]].describe().loc[["mean", "std"]].round(1))

fe = FeatureEngineer()
fe.transform(X_train.head())[["month", "next_month_sin", "next_month_cos", "next_is_summer",
                              "units_per_person", "ac_x_temp"]]

# Assemble the full preprocessing pipeline (Steps 7-10)
numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
onehot_pipe  = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
ordinal_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OrdinalEncoder(categories=HOUSE_ORDER,
                                                   handle_unknown="use_encoded_value", unknown_value=-1)),
                         ("scale", StandardScaler())])

def make_preprocessor():
    ct = ColumnTransformer([("num", numeric_pipe, NUM_COLS),
                            ("ohe", onehot_pipe, CAT_ONEHOT),
                            ("ord", ordinal_pipe, CAT_ORDINAL)], remainder="drop")
    return Pipeline([("features", FeatureEngineer()), ("prep", ct)])

prep = make_preprocessor().fit(X_train)
Xt_train = pd.DataFrame(prep.transform(X_train), columns=prep[-1].get_feature_names_out())
print("Shape after preprocessing:", Xt_train.shape, "| any NaN left?", Xt_train.isna().any().any())
Xt_train.describe().loc[["mean", "std"]].round(2)

mi = pd.Series(mutual_info_regression(Xt_train, y_train, random_state=RANDOM_STATE),
               index=Xt_train.columns).sort_values()
mi.plot.barh(figsize=(7, 6), title="Mutual information with next_month_bill"); plt.tight_layout(); plt.show()

# payment_mode check (one-hot, train only)
pm = pd.get_dummies(train.payment_mode.fillna("missing"))
pm_mi = mutual_info_regression(pm, y_train, discrete_features=True, random_state=RANDOM_STATE)
print("payment_mode MI:", dict(zip(pm.columns, pm_mi.round(3))), "-> ~0, so it is dropped")

def evaluate(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
            "MAPE_%": 100 * mean_absolute_percentage_error(y_true, y_pred)}

def make_model(estimator):
    return Pipeline([("prep", make_preprocessor()), ("model", estimator)])

print("Naive baseline (next = current bill):", {k: round(v, 3) for k, v in evaluate(y_val, X_val.bill_amount).items()})
lr = make_model(LinearRegression()).fit(X_train, y_train)
print("Linear Regression:                  ", {k: round(v, 3) for k, v in evaluate(y_val, lr.predict(X_val)).items()})

def time_folds(months, n_folds=4, val_months=2):
    """Expanding-window CV: each fold validates on later months than it trains on."""
    months = months.reset_index(drop=True)
    uniq = sorted(months.unique())
    folds = []
    for k in range(n_folds, 0, -1):
        cut = len(uniq) - k * val_months
        tr = months.index[months.isin(uniq[:cut])].values
        va = months.index[months.isin(uniq[cut:cut + val_months])].values
        folds.append((tr, va))
    return folds

X_train = X_train.reset_index(drop=True); y_train = y_train.reset_index(drop=True)
cv = time_folds(X_train.month)

candidates = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}
cv_rows = []
for name, est in candidates.items():
    s = cross_validate(make_model(est), X_train, y_train, cv=cv,
                       scoring=["neg_mean_absolute_error", "r2"])
    cv_rows.append({"model": name, "CV_MAE": -s["test_neg_mean_absolute_error"].mean(),
                    "CV_MAE_std": s["test_neg_mean_absolute_error"].std(), "CV_R2": s["test_r2"].mean()})
cv_results = pd.DataFrame(cv_rows).sort_values("CV_MAE").round(3)
cv_results

search_spaces = {
    "Random Forest": (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 8, 12, 16],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": [0.5, 0.7, 1.0]}),
    "Gradient Boosting": (HistGradientBoostingRegressor(random_state=RANDOM_STATE), {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_iter": [200, 400, 600],
        "model__max_depth": [None, 4, 6, 8],
        "model__min_samples_leaf": [10, 20, 40],
        "model__l2_regularization": [0.0, 0.1, 1.0]}),
}
tuned = {}
for name, (est, grid) in search_spaces.items():
    rs = RandomizedSearchCV(make_model(est), grid, n_iter=12, cv=cv, scoring="neg_mean_absolute_error",
                            random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
    tuned[name] = rs.best_estimator_
    print(f"{name}: best CV MAE = {-rs.best_score_:.2f}  params = {rs.best_params_}")

final_candidates = {"Naive baseline": None,
                    "Linear Regression": make_model(LinearRegression()).fit(X_train, y_train),
                    "Ridge": make_model(Ridge(alpha=1.0)).fit(X_train, y_train),
                    **{f"{k} (tuned)": v for k, v in tuned.items()}}
rows = []
for name, m in final_candidates.items():
    pred = X_val.bill_amount if m is None else m.predict(X_val)
    rows.append({"model": name, **evaluate(y_val, pred)})
comparison = pd.DataFrame(rows).sort_values("MAE").round(3)
display(comparison)
best_name = comparison[comparison.model != "Naive baseline"].iloc[0].model
best_model = final_candidates[best_name]
print("Selected model:", best_name)

X_trval = pd.concat([X_train, X_val]); y_trval = pd.concat([y_train, y_val])
best_model.fit(X_trval, y_trval)
test_pred = best_model.predict(X_test)
test_metrics = evaluate(y_test, test_pred)
print("TEST:", {k: round(v, 3) for k, v in test_metrics.items()})
print("Naive baseline on TEST:", {k: round(v, 3) for k, v in evaluate(y_test, X_test.bill_amount).items()})

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].scatter(y_test, test_pred, s=8, alpha=.5)
lim = [0, max(y_test.max(), test_pred.max())]; ax[0].plot(lim, lim, "r--")
ax[0].set(xlabel="Actual (₹)", ylabel="Predicted (₹)", title="Actual vs predicted — test")
(test_pred - y_test).plot.hist(bins=40, ax=ax[1], title="Residuals (pred − actual)")
plt.tight_layout(); plt.show()

pi = permutation_importance(best_model, X_test, y_test, scoring="neg_mean_absolute_error",
                            n_repeats=5, random_state=RANDOM_STATE)
importance = pd.Series(pi.importances_mean, index=X_test.columns).sort_values()
importance.plot.barh(figsize=(7, 5), title="Permutation importance (MAE increase, ₹)"); plt.tight_layout(); plt.show()

X_all, y_all = df[INPUT_COLS], df[TARGET]
final_model = best_model.fit(X_all, y_all)
print("Final model trained on", len(X_all), "rows")

artifact = {
    "model": final_model,
    "model_name": best_name,
    "input_cols": INPUT_COLS,
    "test_metrics": test_metrics,
    "comparison": comparison,
    "cv_results": cv_results,
    "importance": importance,
    "test_results": pd.DataFrame({"month": X_test.month.values, "actual": y_test.values, "predicted": test_pred}),
    "train_stats": X_trval[["avg_temp_c", "units_consumed", "bill_amount"]].describe().loc[["mean", "std"]],
    "trained_on": f"{df.month.min()} to {df.month.max()}",
}
joblib.dump(artifact, "bill_model.joblib")
loaded = joblib.load("bill_model.joblib")
print("Saved & reloaded. Sample prediction:", loaded["model"].predict(X_test.head(1)).round(2))

# What the app does for one customer:
latest = df[df.customer_id == "C1000"].sort_values("month").tail(1)
print("Current bill:", latest.bill_amount.item(), "| Predicted next bill:", final_model.predict(latest[INPUT_COLS]).round(2))

res = artifact["test_results"].assign(abs_err=lambda d: (d.actual - d.predicted).abs())
print("MAE by month on test:\n", res.groupby("month").abs_err.mean().round(2))

def psi(expected, actual, bins=10):
    """Population Stability Index: <0.1 stable, 0.1-0.25 moderate drift, >0.25 major drift."""
    expected, actual = expected.dropna(), actual.dropna()
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    e = np.histogram(expected, cuts)[0] / len(expected) + 1e-6
    a = np.histogram(actual.clip(cuts[0], cuts[-1]), cuts)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))

drift = {c: round(psi(X_train[c], X_test[c]), 3) for c in ["avg_temp_c", "humidity_pct", "units_consumed", "bill_amount"]}
print("\nPSI train vs test:", drift)

