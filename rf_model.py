# IMPORT LIBRARIES
import numpy as np
import pandas as pd

# LOAD DATA
df = pd.read_csv("FINAL_MODEL_DEVELOPMENT.csv")

# DEVELOP FEATURE
X = df[[ "TA_F", "VPD_F", "NETRAD", "WS_F" ]]

# DEVELOP TARGET
y = df["ET_mm_day"]

# DATA SPLIT FOR TRAINING AND TESTING
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42)

# IMPORT MODEL[RANDOM FOREST]

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=6,
    min_samples_split=6,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42
)

# MODEL TRAINING
model.fit(X_train, y_train)

# TAKE PREDICTION ON UNSEEN DATA FOR MODEL EVALUATION CHECKING
pre = model.predict(X_test)

# IMPORTING METRICS
from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# CHECKING ERROR VALUES ON UNSEEN DATA
mae = mean_absolute_error(y_test, pre)

mse = mean_squared_error(y_test, pre)

r2 = r2_score(y_test, pre)

# FOR SEEING THESE VALUES ON TERMINAL
print("MAE:", mae)

print("MSE:", mse)

print("R2:", r2)

# TAKE PREDICTION ON TRAINING DATA [WE ENSURE THAT MODEL IS OVERFITTING OR NOT]
train_accu = model.predict(X_train)

# CHECKING ERROR VALUES ON TRAINING DATA
train_mae = mean_absolute_error(y_train, train_accu)

train_mse = mean_squared_error(y_train, train_accu)

train_r2 = r2_score(y_train, train_accu)

# FOR SEEING THESE VALUES ON TERMINAL
print("TRAINING MAE:", train_mae)

print("TRAINING MSE:", train_mse)

print("TRAINING R2:", train_r2)

# WE DO CROSS VALIDATION
from sklearn.model_selection import cross_val_score

Scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("ALL FOLD SCORES:", Scores)
print("AVERAGE SCORE:", Scores.mean())