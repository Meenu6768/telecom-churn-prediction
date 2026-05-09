import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 3333

df = pd.DataFrame({
    "Account length":         np.random.randint(1, 243, n),
    "Area code":              np.random.choice([408, 415, 510], n),
    "International plan":     np.random.choice(["yes", "no"], n, p=[0.1, 0.9]),
    "Voice mail plan":        np.random.choice(["yes", "no"], n, p=[0.3, 0.7]),
    "Number vmail messages":  np.random.randint(0, 51, n),
    "Total day minutes":      np.round(np.random.uniform(0, 350, n), 1),
    "Total day calls":        np.random.randint(0, 165, n),
    "Total day charge":       np.round(np.random.uniform(0, 59, n), 2),
    "Total eve minutes":      np.round(np.random.uniform(0, 363, n), 1),
    "Total eve calls":        np.random.randint(0, 170, n),
    "Total eve charge":       np.round(np.random.uniform(0, 30, n), 2),
    "Total night minutes":    np.round(np.random.uniform(0, 395, n), 1),
    "Total night calls":      np.random.randint(0, 175, n),
    "Total night charge":     np.round(np.random.uniform(0, 17, n), 2),
    "Total intl minutes":     np.round(np.random.uniform(0, 20, n), 1),
    "Total intl calls":       np.random.randint(0, 20, n),
    "Total intl charge":      np.round(np.random.uniform(0, 5, n), 2),
    "Customer service calls": np.random.randint(0, 9, n),
})

churn_prob = (
    (df["Total day minutes"] > 250).astype(int) * 0.3 +
    (df["Customer service calls"] >= 4).astype(int) * 0.4 +
    (df["International plan"] == "yes").astype(int) * 0.2
)
churn_prob = np.clip(churn_prob, 0.05, 0.85)
df["Churn"] = np.random.binomial(1, churn_prob)

# Save to same folder as this script
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telecom_churn.csv")
df.to_csv(save_path, index=False)
print(f"Saved to: {save_path}")
print(f"Records: {len(df)}")
print(f"Churn rate: {df['Churn'].mean()*100:.1f}%")