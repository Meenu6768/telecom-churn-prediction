
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

# Load data
df = pd.read_csv("telecom_churn.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
print("\nTraining Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Save model
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved as churn_model.pkl")

# Plot 1 — Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
top_features = [X.columns[i] for i in indices]
top_scores   = [importances[i] for i in indices]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Telecom Churn Prediction — Model Report", fontsize=14, fontweight="bold")

axes[0].barh(top_features[::-1], top_scores[::-1], color="#1A56A5")
axes[0].set_title("Top 10 Churn Indicators (Feature Importance)")
axes[0].set_xlabel("Importance Score")

# Plot 2 — Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
axes[1].set_title("Confusion Matrix")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("churn_report.png", dpi=150)
plt.show()
print("Report saved as churn_report.png")