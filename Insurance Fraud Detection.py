# 📦 Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, roc_auc_score
)

# 🤖 Machine Learning Models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# 📂 Load and Clean Data
df = pd.read_csv(r"C:\Users\basil\OneDrive\Desktop\Base\Work\Personal projects\autoinsurance\insurance_claims.csv")
df = df.replace('?', np.nan)  # Replace '?' with NaN

# 📅 Convert Dates to datetime format
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df['incident_date'] = pd.to_datetime(df['incident_date'])

# 🧠 Feature Engineering (Time-based features)
df['policy_year'] = df['policy_bind_date'].dt.year
df['policy_month'] = df['policy_bind_date'].dt.month
df['incident_year'] = df['incident_date'].dt.year
df['incident_month'] = df['incident_date'].dt.month
df['days_policy_to_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days

# 🧹 Drop unnecessary columns
df = df.drop(columns=['_c39', 'policy_number', 'age', 'total_claim_amount', 'policy_bind_date', 'incident_date'])

# 🧼 Handle Missing Values
for col in ['collision_type', 'property_damage', 'police_report_available']:
    df[col] = df[col].fillna(df[col].mode()[0])

# 🎯 Encode Target
df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})

# 📉 Visualize Missing Data
msno.bar(df)
plt.title("📊 Missing Values After Cleaning")
plt.show()

# 🔥 Correlation Heatmap
num_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(14, 10))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("🔥 Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()

# 📦 Features & Target
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

# 🏷️ One-Hot Encode Categorical Variables
X_encoded = pd.get_dummies(X, drop_first=True)

# ✂️ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=42, stratify=y
)

# ⚖️ Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🤖 Initialize Models
models = {
    'Support Vector': SVC(random_state=42, probability=True),
    'K-Neighbors': KNeighborsClassifier(n_neighbors=30),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(criterion='entropy', max_depth=10, n_estimators=140, random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

# 🧪 Train and Evaluate Models
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba)
    })

# 📊 Results DataFrame
results_df = pd.DataFrame(results)
print(results_df.sort_values('ROC AUC', ascending=False))

# 🌲 Feature Importance - Random Forest
rf = models['Random Forest']
importances = pd.Series(rf.feature_importances_, index=X_encoded.columns)
plt.figure(figsize=(12, 8))
importances.nlargest(15).plot(kind='barh')
plt.title('🌟 Top 15 Important Features - Random Forest')
plt.show()

# 🔍 Confusion Matrix - Best Model
best_model_name = results_df.loc[results_df['ROC AUC'].idxmax(), 'Model']
best_model = models[best_model_name]
y_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.title(f'Confusion Matrix - {best_model.__class__.__name__}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()



# Example: Predicting Fraud for New CSV Input
# 📥 Load New Data (must match structure of training data)
new_data = pd.read_csv(r"C:\Users\basil\OneDrive\Desktop\Base\Work\Personal projects\autoinsurance\new_claims.csv")
new_data = new_data.replace('?', np.nan)

# 🧹 Preprocessing (same as training data)
new_data['policy_bind_date'] = pd.to_datetime(new_data['policy_bind_date'])
new_data['incident_date'] = pd.to_datetime(new_data['incident_date'])

new_data['policy_year'] = new_data['policy_bind_date'].dt.year
new_data['policy_month'] = new_data['policy_bind_date'].dt.month
new_data['incident_year'] = new_data['incident_date'].dt.year
new_data['incident_month'] = new_data['incident_date'].dt.month
new_data['days_policy_to_incident'] = (new_data['incident_date'] - new_data['policy_bind_date']).dt.days

new_data = new_data.drop(columns=['_c39', 'policy_number', 'age', 'total_claim_amount', 'policy_bind_date', 'incident_date'])

for col in ['collision_type', 'property_damage', 'police_report_available']:
    new_data[col] = new_data[col].fillna(df[col].mode()[0])  # Use training mode

# One-hot encode using training columns
new_data_encoded = pd.get_dummies(new_data, drop_first=True)

# Match column order with training data
new_data_encoded = new_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Scale features
new_data_scaled = scaler.transform(new_data_encoded)

# Predict using best model
new_predictions = best_model.predict(new_data_scaled)
new_probs = best_model.predict_proba(new_data_scaled)[:, 1]

# 🎉 Show results
output_df = new_data.copy()
output_df['fraud_predicted'] = new_predictions
output_df['fraud_probability'] = new_probs
print(output_df[['fraud_predicted', 'fraud_probability']])