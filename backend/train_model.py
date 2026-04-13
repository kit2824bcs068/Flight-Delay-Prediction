import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

print("=" * 55)
print("  Flight Delay Prediction — Model Training")
print("=" * 55)

DATASET_PATH = "flights.csv"

if os.path.exists(DATASET_PATH):
    print(f"\n[1/5] Loading real dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df = df.sample(n=200_000, random_state=42) if len(df) > 200_000 else df
    print(f"      Loaded {len(df):,} rows")
else:
    print("\n[1/5] Generating synthetic dataset...")
    np.random.seed(42)
    n = 50_000
    airlines  = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"]
    airports  = ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "LAS", "SEA", "MIA"]
    months      = np.random.randint(1, 13, n)
    days        = np.random.randint(1, 8, n)
    dep_hours   = np.random.randint(5, 23, n)
    distances   = np.random.randint(100, 3000, n)
    airline_col = np.random.choice(airlines, n)
    origin_col  = np.random.choice(airports, n)
    dest_col    = np.random.choice(airports, n)
    delay_prob = (
        0.05
        + 0.15 * np.isin(airline_col, ["NK", "F9"]).astype(float)
        + 0.10 * np.isin(months, [12, 1, 2, 6, 7]).astype(float)
        + 0.10 * np.isin(origin_col, ["ORD", "JFK", "SFO"]).astype(float)
        + 0.12 * ((dep_hours >= 16) & (dep_hours <= 20)).astype(float)
        + 0.08 * np.isin(days, [5, 7]).astype(float)
        + np.random.uniform(0, 0.2, n)
    )
    delay_prob = np.clip(delay_prob, 0, 1)
    delayed = (np.random.rand(n) < delay_prob).astype(int)
    df = pd.DataFrame({
        "AIRLINE": airline_col,
        "MONTH": months,
        "DAY_OF_WEEK": days,
        "SCHEDULED_DEPARTURE": dep_hours * 100,
        "ORIGIN_AIRPORT": origin_col,
        "DESTINATION_AIRPORT": dest_col,
        "DISTANCE": distances,
        "DEPARTURE_DELAY": np.where(delayed, np.random.randint(15, 200, n), 0),
    })
    print(f"      Generated {len(df):,} rows | {delayed.mean():.1%} delayed")

print("\n[2/5] Engineering features...")
df["DEP_HOUR"]     = (df["SCHEDULED_DEPARTURE"] // 100).astype(int)
df["DELAYED"]      = (df["DEPARTURE_DELAY"] > 15).astype(int)
df["IS_WEEKEND"]   = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)
df["IS_PEAK_HOUR"] = df["DEP_HOUR"].between(16, 20).astype(int)
df["IS_WINTER"]    = df["MONTH"].isin([12, 1, 2]).astype(int)
df["IS_SUMMER"]    = df["MONTH"].isin([6, 7, 8]).astype(int)
df = pd.get_dummies(df, columns=["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], prefix=["AIRLINE", "ORIGIN", "DEST"])

print("\n[3/5] Preparing training data...")
drop_cols = ["DEPARTURE_DELAY", "SCHEDULED_DEPARTURE"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
df.dropna(subset=["DELAYED"], inplace=True)
df.fillna(0, inplace=True)

X = df.drop(columns=["DELAYED"])
y = df["DELAYED"]
feature_columns = list(X.columns)
print(f"      Features: {len(feature_columns)} | Samples: {len(X):,}")
print(f"      Delay rate: {y.mean():.1%}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

print("\n[4/5] Training Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=150, max_depth=12,
    min_samples_split=10, min_samples_leaf=5,
    n_jobs=-1, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)
print("      Training complete!")

print("\n[5/5] Evaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n  Accuracy: {acc*100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=["On Time", "Delayed"]))

joblib.dump({"model": model, "feature_columns": feature_columns}, "flight_delay_model.pkl")
print("\n✅ Model saved to flight_delay_model.pkl")
print("   Now run: uvicorn main:app --reload")