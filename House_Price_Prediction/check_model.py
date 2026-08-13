# import os
# import joblib

# # Load the existing trained model
# model = joblib.load("house_price_model.pkl")

# # Check model parameters
# print("Number of trees:", model.n_estimators)
# print("Max depth:", model.max_depth)

# # Check file size
# size_mb = os.path.getsize("house_price_model.pkl") / (1024 * 1024)

# print("Model file size:", size_mb, "MB")


import os

size_mb = os.path.getsize(
    "house_price_model_small.pkl"
) / (1024 * 1024)

print(size_mb)
