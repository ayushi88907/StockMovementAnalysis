# import pandas as pd
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import nltk

# nltk.download("vader_lexicon")

# # Load_df = pd.read_csv("./data/raw_data.csv") data
# tweets

# # Sentiment analysis
# sia = SentimentIntensityAnalyzer()
# tweets_df["timenSent"] = tweets_df["text"].apply(lambda x: sia.polarity_scores(x)["compound"])

# # Save processed data
# tweets_df.to_csv("./data/processed_data.csv", index=False)
# print("Processed data saved to ./data/processed_data.csv")


import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download necessary NLTK resources
nltk.download("vader_lexicon")
nltk.download("stopwords")

# Load data
tweets_df = pd.read_csv("./data/raw_data.csv")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)    # Remove mentions
    text = re.sub(r"#", "", text)       # Remove hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Remove special characters
    text = text.lower()                # Convert to lowercase
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text

# Clean the text
tweets_df["Cleaned_Text"] = tweets_df["text"].apply(clean_text)

# Perform sentiment analysis
tweets_df["Sentiment"] = tweets_df["Cleaned_Text"].apply(lambda x: sia.polarity_scores(x)["compound"])

# Extract frequency of mentions for keywords like "stock" or "market"
keywords = ["stock", "market", "investing", "trade"]
for keyword in keywords:
    tweets_df[f"Freq_{keyword}"] = tweets_df["Cleaned_Text"].str.count(keyword)

# Save processed data
tweets_df.to_csv("./data/processed_data.csv", index=False)
print("Processed data saved to ./data/processed_data.csv")
