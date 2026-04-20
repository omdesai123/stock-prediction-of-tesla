"""
model.py
Tesla (TSLA) Stock Movement Predictor
--------------------------------------
Fetches historical data, engineers features, trains a RandomForest classifier,
evaluates it, and saves the trained model to model.pkl.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ─────────────────────────────────────────────
# 1. FETCH DATA
# ─────────────────────────────────────────────
print("Fetching TSLA historical data...")
df = yf.download("TSLA", start="2018-01-01", auto_adjust=True)
df.dropna(inplace=True)

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
# Moving averages
df["MA10"]  = df["Close"].rolling(window=10).mean()
df["MA50"]  = df["Close"].rolling(window=50).mean()

# Daily return: percentage change from previous close
df["Daily_Return"] = df["Close"].pct_change()

# 7-day average volume
df["Avg_Vol_7"] = df["Volume"].rolling(window=7).mean()

# MA ratio: relative position of short MA vs long MA
df["MA_Ratio"] = df["MA10"] / df["MA50"]

# Target: 1 if tomorrow's close > today's close, else 0
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Drop rows with NaN values introduced by rolling windows
df.dropna(inplace=True)

# ─────────────────────────────────────────────
# 3. PREPARE FEATURES & SPLIT
# ─────────────────────────────────────────────
FEATURE_COLS = ["MA10", "MA50", "Daily_Return", "Avg_Vol_7", "MA_Ratio"]

X = df[FEATURE_COLS]
y = df["Target"]

# 80/20 chronological split (no shuffle to prevent data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# ─────────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────────
print("Training RandomForest classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 5. EVALUATE MODEL
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print(f"\nTest Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
print("Confusion Matrix :")
print(cm)

# ─────────────────────────────────────────────
# 6. SAVE MODEL
# ─────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to model.pkl ✓")
