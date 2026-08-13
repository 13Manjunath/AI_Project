# House price prediction project

#  we are using Linear regression, random forest, xgboost

# step 1 --> import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
from fastapi import FastAPI


# load the dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(df.head())
print("shape is",df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.columns)

# step 2 --> EDA
plt.figure(figsize=(10, 8))
plt.hist(df['MedHouseVal'], bins=30)
plt.xlabel("House Value")
plt.ylabel("Frequency")
plt.title("Distribution of House Values" )
# plt.show()

corr = df.corr()
print(corr["MedHouseVal"].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
plt.scatter(df['MedInc'], df["MedHouseVal"], alpha=0.3)
plt.xlabel("Median Income")
plt.ylabel("House value")
plt.title("Med income vs House Value")
# plt.show()

df.hist(figsize=(10, 8), bins = 30)
plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 8))
df.boxplot()
plt.xticks(rotation = 45)
plt.title("Feature Outliers")
# plt.show()

# step 3 --> Data preprocessing and train/test split
X = df.drop("MedHouseVal", axis = 1)
Y = df["MedHouseVal"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Y_train :", Y_train.shape)
print("Y_test: ", Y_test.shape)

scaler = StandardScaler()
X_trained_scale = scaler.fit_transform(X_train)
X_tested_scale = scaler.transform(X_test)

print("Training mean :", X_trained_scale.mean(axis=0))
print("Training std:", X_trained_scale.std(axis=0))


# step 4 --> Model Training

# Train Linear model
lr_model = LinearRegression()
lr_model.fit(X_trained_scale, Y_train)

Y_predict_lr = lr_model.predict(X_tested_scale)

mse = mean_squared_error(Y_test, Y_predict_lr)
mae = mean_absolute_error(Y_test, Y_predict_lr)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_predict_lr)

print("Linear Regression")
print("Mae : ", mae)
print("mse :", mse)
print("rmse:", rmse)
print("R* R", r2)

comparision = pd.DataFrame({"Actual":Y_test.values, "Predicted":Y_predict_lr})
print(comparision.head(10))

# Train the random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1) 
rf_model.fit(X_train, Y_train)

Y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(Y_test, Y_pred_rf)
mse_rf = mean_squared_error(Y_test, Y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(Y_test, Y_pred_rf)
print("Random Forest")
print("Mae :", mae_rf)
print("mse:", mse_rf)
print("rmse:", rmse_rf)
print("r2_score", r2_rf)

result = pd.DataFrame({"Model":["Linear Regression", "Random Forest"], "MAE":[mae, mae_rf], "MSE":[mse, mse_rf], "RMSE": [rmse, rmse_rf], "R2": [r2, r2_rf]
})

print(result)

# train the gradient boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate= 0.1, random_state=42, max_depth=3)
gb_model.fit(X_train, Y_train)

Y_pred_gb = gb_model.predict(X_test)

mae_gb = mean_absolute_error(Y_test, Y_pred_gb)
mse_gb = mean_squared_error(Y_test, Y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(Y_test, Y_pred_gb)

print("Gradient Boosting")
print("MAE :", mae_gb)
print("MSE :", mse_gb)
print("RMSE:", rmse_gb)
print("R²  :", r2_gb)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],
    "MAE": [
        mae,
        mae_rf,
        mae_gb
    ],
    "MSE": [
        mse,
        mse_rf,
        mse_gb
    ],
    "RMSE": [
        rmse,
        rmse_rf,
        rmse_gb
    ],
    "R2": [
        r2,
        r2_rf,
        r2_gb
    ]
})

print(results)

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
xgb_model.fit(X_train, Y_train)
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(Y_test, y_pred_xgb)

mse_xgb = mean_squared_error(Y_test, y_pred_xgb)

rmse_xgb = np.sqrt(mse_xgb)

r2_xgb = r2_score(Y_test, y_pred_xgb)

print("XGBoost")
print("MAE :", mae_xgb)
print("MSE :", mse_xgb)
print("RMSE:", rmse_xgb)
print("R²  :", r2_xgb)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ],
    "MAE": [
        mae,
        mae_rf,
        mae_gb,
        mae_xgb
    ],
    "MSE": [
        mse,
        mse_rf,
        mse_gb,
        mse_xgb
    ],
    "RMSE": [
        rmse,
        rmse_rf,
        rmse_gb,
        rmse_xgb
    ],
    "R2": [
        r2,
        r2_rf,
        r2_gb,
        r2_xgb
    ]
})

print(results)


lgb_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    verbosity=-1
)

lgb_model.fit(X_train, Y_train)
y_pred_lgb = lgb_model.predict(X_test)
mae_lgb = mean_absolute_error(Y_test, y_pred_lgb)

mse_lgb = mean_squared_error(Y_test, y_pred_lgb)

rmse_lgb = np.sqrt(mse_lgb)

r2_lgb = r2_score(Y_test, y_pred_lgb)

print("LightGBM")
print("MAE :", mae_lgb)
print("MSE :", mse_lgb)
print("RMSE:", rmse_lgb)
print("R²  :", r2_lgb)

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM"
    ],
    "MAE": [
        mae,
        mae_rf,
        mae_gb,
        mae_xgb,
        mae_lgb
    ],
    "MSE": [
        mse,
        mse_rf,
        mse_gb,
        mse_xgb,
        mse_lgb
    ],
    "RMSE": [
        rmse,
        rmse_rf,
        rmse_gb,
        rmse_xgb,
        rmse_lgb
    ],
    "R2": [
        r2,
        r2_rf,
        r2_gb,
        r2_xgb,
        r2_lgb
    ]
})

print(results.sort_values("R2", ascending=False))

# step 5 Cross validation

# rf_cv_scores = cross_val_score(
#     rf_model,
#     X_train,
#     Y_train,
#     cv=5,
#     scoring="r2",
#     n_jobs=-1
# )

# print("Random Forest CV scores:", rf_cv_scores)
# print("Mean CV R²:", rf_cv_scores.mean())
# print("Std CV R²:", rf_cv_scores.std())

# models = {
#     "Random Forest": rf_model,
#     "Gradient Boosting": gb_model,
#     "XGBoost": xgb_model,
#     "LightGBM": lgb_model
# }

# cv_results = []

# for name, model in models.items():

#     scores = cross_val_score(
#         model,
#         X_train,
#         Y_train,
#         cv=5,
#         scoring="r2",
#         n_jobs=-1
#     )

#     cv_results.append({
#         "Model": name,
#         "Mean R2": scores.mean(),
#         "Std R2": scores.std()
#     })

#     print(name)
#     print("CV Scores:", scores)
#     print("Mean R²:", scores.mean())
#     print("Std R²:", scores.std())
#     print("-" * 40)
    
#     cv_results_df = pd.DataFrame(cv_results)

# # print(
#     cv_results_df.sort_values(
#         "Mean R2",
#         ascending=False
#     )


# # Step --6 Hyper parameter Tuning
# RandomForestRegressor(
#     n_estimators=100,
#     random_state=42
# )
# rf = RandomForestRegressor(
#     random_state=42,
#     n_jobs=-1
# )

# param_grid = {
#     "n_estimators": [100, 200, 300, 500],
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "max_features": [1.0, "sqrt", "log2"]
# }
# rf_random_search = RandomizedSearchCV(
#     estimator=rf,
#     param_distributions=param_grid,
#     n_iter=20,
#     cv=5,
#     scoring="r2",
#     random_state=42,
#     n_jobs=-1,
#     verbose=1
# )

# rf_random_search.fit(X_train, Y_train)

# print("Best Parameters:")
# print(rf_random_search.best_params_)

# print("\nBest CV R²:")
# print(rf_random_search.best_score_)

# best_rf = rf_random_search.best_estimator_

# y_pred_tuned = best_rf.predict(X_test)

# mae_tuned = mean_absolute_error(Y_test, y_pred_tuned)
# mse_tuned = mean_squared_error(Y_test, y_pred_tuned)
# rmse_tuned = np.sqrt(mse_tuned)
# r2_tuned = r2_score(Y_test, y_pred_tuned)

# print("Tuned Random Forest")
# print("MAE :", mae_tuned)
# print("MSE :", mse_tuned)
# print("RMSE:", rmse_tuned)
# print("R²  :", r2_tuned)

# comparison = pd.DataFrame({
#     "Model": [
#         "Random Forest Before Tuning",
#         "Random Forest After Tuning"
#     ],
#     "RMSE": [
#         rmse_rf,
#         rmse_tuned
#     ],
#     "R2": [
#         r2_rf,
#         r2_tuned
#     ]
# })

# print(comparison)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

small_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

small_rf.fit(X_train, Y_train)

y_pred = small_rf.predict(X_test)

print("R²:", r2_score(Y_test, y_pred))

# joblib.dump(
#     small_rf,
#     "house_price_model_small.pkl"
# )
# from sklearn.metrics import r2_score

# y_pred = small_rf.predict(X_test)

# r2 = r2_score(Y_test, y_pred)

# print("R²:", r2)
