import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model and data
with open("./src/stock_model.pkl", "rb") as f:
    model = pickle.load(f)

data = pd.read_csv("./data/processed_data.csv")
data["Predicted_Change"] = model.predict(data[["Sentiment"]])

# Plot
data["created_at"] = pd.to_datetime(data["created_at"])
data = data.sort_values("created_at")

plt.figure(figsize=(10, 6))
plt.plot(data["created_at"], data["Predicted_Change"], label="Predicted Stock Change")
plt.xlabel("Time")
plt.ylabel("Stock Movement (0 = Down, 1 = Up)")
plt.legend()
plt.show()
plt.savefig("./data/visualization_output.png")

