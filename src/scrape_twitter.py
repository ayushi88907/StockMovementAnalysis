import os
import pandas as pd
import requests
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Twitter API v2 headers
HEADERS = {"Authorization": f"Bearer {BEARER_TOKEN}"}

# Function to fetch tweets
def fetch_tweets_v2(query, max_results=10):
    url = f"https://api.twitter.com/2/tweets/search/recent?query=%23StockMarket%20OR%20%23Trading&tweet.fields=created_at,lang&max_results={max_results}"
    # url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&tweet.fields=created_at,lang&max_results={max_results}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code}, {response.json()}")

    tweets = response.json().get("data", [])
    return pd.DataFrame(tweets)

if __name__ == "__main__":
    query = "#StockMarket OR #Trading"  # A valid query string
    max_results = 80  # Adjust as needed (max 100)
    try:
        tweets_df = fetch_tweets_v2(query, max_results)
        tweets_df.to_csv("./data/raw_data.csv", index=False)
        print("Tweets saved to ./data/raw_data.csv")
        print(tweets_df.head())
        print(tweets_df.columns)
    except Exception as e:
        print("Error occurred:", e)
