# Stock Movement Prediction Using Social Media Sentiment Analysis

## 1. Project Overview
This project predicts stock movements using social media sentiment analysis. It scrapes tweets, processes them for sentiment, and trains a machine-learning model to classify stock movements.

## 2. Prerequisites
- Python Version: 3.8+
- Dependencies: Listed in `requirements.txt`.

## 3. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd StockPredictionProject
```
### Step 2: Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # For macOS/Linux
venv\\Scripts\\activate      # For Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Keys
  1. Create a `.env` file in the root directory.
  2. Add the following keys (replace with actual values):

```bash
BEARER_TOKEN=<Your_Twitter_API_Bearer_Token>
```

## 4. How to Run the Code
### Step 1: Scrape Tweets

```bash
python src/scrape_twitter.py
```
This will save raw data to ./data/raw_data.csv.

### Step 2: Preprocess Data

```bash
python src/preprocess.py
```
This will generate processed_data.csv with sentiment scores.

### Step 3: Train the Model

```bash
python src/train_model.py
```
The trained model will be saved as stock_model.pkl.

### Step 4: Visualize Predictions

```bash
python src/visualize.py

```
This will display stock sentiment trends and predictions graphically.


## 5. Project Structure

```bash
StockPredictionProject/
├── data/
│   ├── raw_data.csv            # Raw data from scraping
│   ├── processed_data.csv      # Data after preprocessing
├── src/
│   ├── scrape_twitter.py       # Data scraping script
│   ├── preprocess.py           # Preprocessing script
│   ├── train_model.py          # Model training script
│   ├── visualize.py            # Visualization script
├── requirements.txt            # Python dependencies
├── README.md                   # Project instructions
├── report.pdf                  # Project report
└── .env                        # API keys (not included in repo)

```

## 6. Notes
  Ensure API rate limits are handled during scraping.
  Results may vary depending on the size and diversity of the dataset.


