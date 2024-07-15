import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = file_path = r'C:\Users\Sree Murugan S J\Desktop\creditcard.csv' # Replace with the actual file path
data = pd.read_csv(file_path)

# Check for missing values
print(data.isnull().sum())

# Check the distribution of the target variable
print(data['Class'].value_counts())

# Separate the features and the target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train the Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict the classes
y_train_pred_rf = rf_classifier.predict(X_train)
y_test_pred_rf = rf_classifier.predict(X_test)

# Evaluate the model
print("Training Data Evaluation (Random Forest):")
print(classification_report(y_train, y_train_pred_rf))
print(confusion_matrix(y_train, y_train_pred_rf))
print("Accuracy:", accuracy_score(y_train, y_train_pred_rf))

print("\nTesting Data Evaluation (Random Forest):")
print(classification_report(y_test, y_test_pred_rf))
print(confusion_matrix(y_test, y_test_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_test_pred_rf))
