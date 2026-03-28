# sentiment/sentiment_analyzer.py
import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Fetch news and compute sentiment scores."""
    def __init__(self, ticker='INFY.NS', use_hf=False):
        self.ticker = ticker
        self.analyzer = SentimentIntensityAnalyzer()
        self.use_hf = use_hf
        if use_hf:
            try:
                from transformers import pipeline
                self.hf_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
            except ImportError:
                logger.warning("Transformers not installed, falling back to VADER")
                self.use_hf = False

    def fetch_news(self, days_back=7):
        """Fetch news headlines for the ticker from Yahoo Finance RSS."""
        # Yahoo Finance RSS uses ticker symbol without .NS suffix
        symbol = self.ticker.split('.')[0]
        feed_url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US'
        feed = feedparser.parse(feed_url)
        news_items = []
        cutoff = datetime.now() - timedelta(days=days_back)
        for entry in feed.entries:
            published = datetime(*entry.published_parsed[:6])
            if published < cutoff:
                continue
            news_items.append({
                'date': published.date(),
                'title': entry.title,
                'summary': entry.summary
            })
        return news_items

    def get_sentiment_scores(self, texts):
        """Compute sentiment scores for a list of texts."""
        scores = []
        for text in texts:
            if self.use_hf:
                result = self.hf_pipeline(text[:512])[0]
                # map to compound-like scale -1 to 1
                score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            else:
                score = self.analyzer.polarity_scores(text)['compound']
            scores.append(score)
        return scores

    def get_daily_sentiment(self, days_back=7):
        """Return a DataFrame with daily average sentiment."""
        news = self.fetch_news(days_back)
        if not news:
            logger.warning("No news fetched.")
            return pd.DataFrame(columns=['date', 'sentiment'])

        df_news = pd.DataFrame(news)
        # Combine title and summary for analysis
        df_news['text'] = df_news['title'] + ' ' + df_news['summary']
        df_news['sentiment'] = self.get_sentiment_scores(df_news['text'])
        daily_sent = df_news.groupby('date')['sentiment'].mean().reset_index()
        return daily_sent