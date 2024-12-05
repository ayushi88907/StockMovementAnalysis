# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import pickle

# # Load data
# data = pd.read_csv("./data/processed_data.csv")

# # Create target column (simulated for demo purposes)
# data["Stock_Change"] = (data["timenSent"] > 0).astype(int)

# # Train-test split
# X = data[["timenSent"]]
# y = data["Stock_Change"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate and save model
# y_pred = model.predict(X_test)
# print("Model Accuracy:", accuracy_score(y_test, y_pred))

# with open("./src/stock_model.pkl", "wb") as f:
#     pickle.dump(model, f)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load data
data = pd.read_csv("./data/processed_data.csv")

# Create target column (simulated for demo purposes)
data["Stock_Change"] = (data["Sentiment"] > 0).astype(int)

# Train-test split
X = data[["Sentiment"]]  # Features
y = data["Stock_Change"]  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Classification report
report = classification_report(y_test, y_pred, target_names=["Down", "Up"])
print("Classification Report:\n", report)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model
with open("./src/stock_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as ./src/stock_model.pkl")
