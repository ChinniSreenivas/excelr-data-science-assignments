# -----------------------------------------------------------
# 📘 Drug Response Classification using Support Vector Machine (SVM)
# -----------------------------------------------------------
# Objective:
# Predict whether a patient has a positive (1) or no (0) response to a drug
# using machine learning techniques, particularly the Support Vector Machine.

# -----------------------------------------------------------
# 1️⃣ Importing Required Libraries
# -----------------------------------------------------------
import pandas as pd               # For handling dataset
import numpy as np                # For numerical operations
import matplotlib.pyplot as plt   # For data visualization
import seaborn as sns             # For statistical plots

# Scikit-learn modules for model building
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# -----------------------------------------------------------
# 2️⃣ Load the Dataset
# -----------------------------------------------------------
# Read the Pharma Industry dataset (make sure the file is in the same folder)
df = pd.read_csv("Pharma_Industry.csv")

# Display dataset shape and column names
print("🔹 Dataset Shape:", df.shape)
print("🔹 Columns:", df.columns.tolist())

# Display first few records
print("\n🔹 First 5 Rows:")
print(df.head())

# -----------------------------------------------------------
# 3️⃣ Exploratory Data Analysis (EDA)
# -----------------------------------------------------------

# Summary statistics (mean, std, min, max, etc.)
print("\n🔹 Summary Statistics:")
print(df.describe())

# Check for missing values
print("\n🔹 Missing Values:\n", df.isnull().sum())

# Visualize class distribution (Target Variable)
plt.figure(figsize=(5,4))
sns.countplot(x='DrugResponse', data=df, palette='coolwarm')
plt.title("Drug Response Distribution (0 = No Response, 1 = Positive Response)")
plt.xlabel("Drug Response")
plt.ylabel("Count")
plt.show()

# Correlation heatmap to identify relationships between features
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Blues')
plt.title("Feature Correlation Heatmap")
plt.show()

# -----------------------------------------------------------
# 4️⃣ Data Preprocessing
# -----------------------------------------------------------

# Encode categorical features if they exist
# (LabelEncoder assigns numeric values to text-based features)
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Separate features (X) and target (y)
X = df.drop('DrugResponse', axis=1)   # Independent variables
y = df['DrugResponse']                # Dependent variable (Target)

# Standardize numerical features for better SVM performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\n✅ Data split completed:")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -----------------------------------------------------------
# 5️⃣ Model Training - Support Vector Machine
# -----------------------------------------------------------
# Initialize SVM model with default 'rbf' kernel
svm_model = SVC(kernel='rbf', random_state=42)

# Train the model on training data
svm_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = svm_model.predict(X_test)

# -----------------------------------------------------------
# 6️⃣ Model Evaluation
# -----------------------------------------------------------
print("\n📊 Model Evaluation Metrics:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix - SVM Model")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# -----------------------------------------------------------
# 7️⃣ Hyperparameter Tuning - Grid Search
# -----------------------------------------------------------
# Define grid of hyperparameters for optimization
param_grid = {
    'C': [0.1, 1, 10],                 # Regularization strength
    'kernel': ['linear', 'poly', 'rbf'],  # Kernel types to test
    'gamma': ['scale', 'auto']         # Kernel coefficient
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,               # 5-fold cross validation
    scoring='accuracy', # Metric for model selection
    n_jobs=-1           # Use all CPU cores
)

# Fit grid search on training data
grid_search.fit(X_train, y_train)

# Display best hyperparameters and score
print("\n🔍 Best Parameters from Grid Search:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Retrain model using best parameters
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test)

# -----------------------------------------------------------
# 8️⃣ Final Model Evaluation
# -----------------------------------------------------------
print("\n⭐ Final Optimized Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# Plot confusion matrix for best model
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Tuned SVM Model")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# -----------------------------------------------------------
# 9️⃣ Visualization of Classification Results (2D)
# -----------------------------------------------------------
# NOTE: For visualization, only first two features are plotted
plt.figure(figsize=(6,5))
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_pred_best, palette='Set2')
plt.title("SVM Classification Results (Test Data)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# -----------------------------------------------------------
# ✅ END OF PROJECT
# -----------------------------------------------------------
# This project demonstrates:
# - Data loading and exploration
# - Preprocessing and feature scaling
# - SVM model training and evaluation
# - Hyperparameter tuning via GridSearchCV
# - Final model performance visualization
