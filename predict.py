import pickle
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

customer = {
    "Account length":         120,
    "Area code":              415,
    "International plan":     1,
    "Voice mail plan":        0,
    "Number vmail messages":  0,
    "Total day minutes":      250.0,
    "Total day calls":        110,
    "Total day charge":       42.5,
    "Total eve minutes":      180.0,
    "Total eve calls":        90,
    "Total eve charge":       15.3,
    "Total night minutes":    200.0,
    "Total night calls":      100,
    "Total night charge":     9.0,
    "Total intl minutes":     10.0,
    "Total intl calls":       4,
    "Total intl charge":      2.7,
    "Customer service calls": 4
}

df = pd.DataFrame([customer])

prediction  = model.predict(df)[0]
probability = model.predict_proba(df)[0]

print("\n--- Churn Prediction ---")
print(f"Prediction   : {'WILL CHURN' if prediction == 1 else 'WILL NOT CHURN'}")
print(f"Churn risk   : {probability[1] * 100:.1f}%")
print(f"Retain chance: {probability[0] * 100:.1f}%")

if probability[1] > 0.6:
    print("\nAction: High risk — offer retention discount immediately.")
elif probability[1] > 0.3:
    print("\nAction: Medium risk — send engagement offer.")
else:
    print("\nAction: Low risk — no action needed.")