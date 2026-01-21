import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Load the Wine Dataset
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['cultivar'] = data.target

# 2. Feature Selection
# We must select exactly 6 features as per instructions.
# Selected Features: Alcohol, Magnesium, Flavanoids, Color Intensity, Hue, Proline
selected_features = ['alcohol', 'magnesium', 'flavanoids', 'color_intensity', 'hue', 'proline']
X = df[selected_features]
y = df['cultivar']

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing & Model Pipeline
# The instructions require Feature Scaling. 
# We use a Pipeline to bundle the Scaler and the Model. This ensures new data in the app is scaled automatically.
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature Scaling (Mandatory)
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)) # Algorithm
])

# 5. Train the Model
print("Training Model...")
pipeline.fit(X_train, y_train)

# 6. Evaluate the Model
y_pred = pipeline.predict(X_test)

print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 7. Save the Model
# Create the model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

model_filename = 'model/wine_cultivar_model.pkl'
joblib.dump(pipeline, model_filename)
print(f"Model saved to {model_filename}")