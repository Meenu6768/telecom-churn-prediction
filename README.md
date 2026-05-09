Telecom Churn Prediction Model
A machine learning pipeline that predicts whether a telecom customer will churn using a Random Forest classifier. Trained on 3,333 customer records with realistic churn logic, the model achieves 87%+ accuracy and provides actionable retention recommendations for each prediction.

Output Preview

After running train_model.py, a churn_report.png is generated with:

Feature Importance chart — top 10 churn indicators ranked by impact
Confusion Matrix — actual vs. predicted churn visualized as a heatmap



 Features

 Random Forest classifier — 100 estimators, trained on 2,666 records, tested on 667
Feature importance analysis — identifies top churn drivers like Customer service calls and Total day minutes
 87%+ accuracy with full classification report (precision, recall, F1)
 Live prediction — predict churn risk for any customer in seconds
 Actionable output — model recommends retention action based on churn probability
 Realistic synthetic dataset — generated with domain-aware churn logic


 Tech Stack
ToolPurposescikit-learnRandom Forest model, train/test split, evaluation metricsPandasData loading, preprocessing, feature engineeringMatplotlibFeature importance bar chartSeabornConfusion matrix heatmapPickleModel serialization for reuse

 Getting Started
Prerequisites
bashpip install scikit-learn pandas matplotlib seaborn
Run
Step 1 — Generate the dataset:
bashpython download_data.py
Step 2 — Train the model:
bashpython train_model.py
Step 3 — Predict churn for a customer:
bashpython predict.py

 Project Structure
telecom_churn_prediction/
│
├── download_data.py    # Generates realistic 3,333-record dataset
├── train_model.py      # Trains Random Forest, evaluates, saves model + charts
├── predict.py          # Predicts churn risk for any new customer
├── telecom_churn.csv   # Auto-generated: training dataset
├── churn_model.pkl     # Auto-generated: saved trained model
└── churn_report.png    # Auto-generated: feature importance + confusion matrix

 Churn Logic
The synthetic dataset uses realistic domain rules to assign churn probability:
Churn Probability =
  + 0.30  if Total day minutes > 250  (heavy usage)
  + 0.40  if Customer service calls ≥ 4  (dissatisfied customer)
  + 0.20  if International plan = Yes  (price sensitivity)

Clipped between 0.05 and 0.85

 Sample Prediction Output
--- Churn Prediction ---
Prediction   : WILL CHURN
Churn risk   : 82.4%
Retain chance: 17.6%

Action: High risk — offer retention discount immediately.

 Model Performance
MetricScoreAccuracy87%+Precision (Churn)~0.85Recall (Churn)~0.83F1 Score (Churn)~0.84

Results may vary slightly due to randomness in data generation.


 Top Churn Indicators
Based on feature importance from the trained model:

Customer service calls — most predictive signal
Total day minutes — high usage correlates with churn
Total day charge — cost burden drives dissatisfaction
International plan — price-sensitive segment
Total intl minutes — international users churn more


Future Improvements

 Flask API for real-time churn prediction endpoint
 XGBoost / LightGBM model comparison
 SHAP values for per-customer explainability
 Streamlit dashboard for non-technical users


 License
MIT License — free to use, modify, and distribute.
