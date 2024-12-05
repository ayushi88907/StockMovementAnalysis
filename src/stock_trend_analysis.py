import os 
import pandas as pd 
import requests 
import re 
from dotenv import load_dotenv 
from nltk.corpus import stopwords 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
import nltk 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pickle

# Check for NLTK resources 
def download_nltk_resources(): 
    try: 
        nltk.data.find('corpora/vader_lexicon') 
    except LookupError: 
        nltk.download("vader_lexicon") 
    try: 
        nltk.data.find('corpora/stopwords') 
    except LookupError: 
        nltk.download("stopwords")
    
download_nltk_resources()

# Load API keys from .env file
load_dotenv()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
HEADERS = {"Authorization": f"Bearer {BEARER_TOKEN}"}


def log_message(message):
    print(message)
    with open("output/log.txt", "a") as log_file:
        log_file.write(message + "\n")


def fetch_tweets_v2(query, max_results=100):
    url = f"https://api.twitter.com/2/tweets/search/recent?query=%23StockMarket%20OR%20%23Trading&tweet.fields=created_at,lang&max_results={max_results}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}, {response.json()}")
    tweets = response.json().get("data", [])
    return pd.DataFrame(tweets)


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    text = " ".join([
        word for word in text.split() if word not in stopwords.words("english")
    ])
    return text


def main():
    try:
        # Process 1: Fetch Tweets
        log_message("Starting Process of Scraping Data 1: Fetch Tweets")
        query = "#StockMarket OR #Trading"
        max_results = 90
        # tweets_df = fetch_tweets_v2(query, max_results)
        # tweets_df.to_csv("./data/raw_data.csv", index=False)
        log_message("Tweets saved to ./data/raw_data.csv\n")
        # log_message(tweets_df.columns)

        # Process 2: Clean and Analyze Tweets
        log_message("Starting Process of Data Cleaning 2: Clean and Analyze Tweets")
        # Load data
        tweets_df = pd.read_csv("./data/raw_data.csv")
        tweets_df["Cleaned_Text"] = tweets_df["text"].apply(clean_text)
        sia = SentimentIntensityAnalyzer()
        tweets_df["Sentiment"] = tweets_df["Cleaned_Text"].apply(
            lambda x: sia.polarity_scores(x)["compound"])
        keywords = ["stock", "market", "investing", "trade"]
        for keyword in keywords:
            tweets_df[f"Freq_{keyword}"] = tweets_df["Cleaned_Text"].str.count(
                keyword)
        tweets_df.to_csv("./data/processed_data.csv", index=False)
        log_message("Processed data saved to ./data/processed_data.csv\n")

        # Process 3: Train Model
        log_message("Starting Process of Model Training 3: Train Model")
        data = pd.read_csv("./data/processed_data.csv")
        data["Stock_Change"] = (data["Sentiment"] > 0).astype(int)
        X = data[["Sentiment"]]
        y = data["Stock_Change"]
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        log_message(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

        log_message(
            f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Down', 'Up'])}"
        )

        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Down", "Up"],
                    yticklabels=["Down", "Up"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("output/confusion_matrix.png")
        plt.show()
        with open("src/stock_model.pkl", "wb") as f:
            pickle.dump(model, f)
        log_message("Model saved as src/stock_model.pkl\n")

        # Process 4: Predict and Plot
        log_message("Starting Process of Prediction 4: Predict and Plot")
        with open("./src/stock_model.pkl", "rb") as f:
            model = pickle.load(f)
        data = pd.read_csv("./data/processed_data.csv")
        data["Predicted_Change"] = model.predict(data[["Sentiment"]])
        data["created_at"] = pd.to_datetime(data["created_at"])
        data = data.sort_values("created_at")
        plt.figure(figsize=(10, 6))
        plt.plot(data["created_at"],
                 data["Predicted_Change"],
                 label="Predicted Stock Change")
        plt.xlabel("Time")
        plt.ylabel("Stock Movement (0 = Down, 1 = Up)")
        plt.legend()
        plt.savefig("output/visualization_output.png")
        plt.show()
        log_message("Prediction plot saved to data/visualization_output.png")

    except Exception as e:
        log_message(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
