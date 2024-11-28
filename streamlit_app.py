import streamlit as st
import requests
import json
__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from openai import OpenAI

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# API key and URLs for Alpha Vantage
api_key = st.secrets["api_keys"]["alpha_vantage"]
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={api_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={api_key}'

# Initialize OpenAI Client
openai_client = OpenAI(api_key=st.secrets["api_keys"]["openai"])

# Streamlit App Title
st.title("Financial Insights Newsletter Generator")

# Sidebar options
option = st.sidebar.radio(
    "Choose an action:",
    ["Load News Data", "Retrieve News Data", "Load Ticker Trends Data", "Retrieve Ticker Trends Data", "Generate Newsletter"]
)

# Initialize ChromaDB Collections
news_collection = client.get_or_create_collection("news_sentiment_data")
ticker_collection = client.get_or_create_collection("ticker_trends_data")

### Function to Load News Data into ChromaDB ###
def load_news_data():
    try:
        response = requests.get(news_url)
        response.raise_for_status()
        data = response.json()

        if 'feed' in data:
            for i, item in enumerate(data['feed']):
                content = item.get("summary", item["title"])
                metadata = {
                    "title": item["title"],
                    "source": item["source"],
                    "time_published": item["time_published"],
                    "overall_sentiment": item.get("overall_sentiment_label", "Neutral")
                }
                # Add to ChromaDB
                news_collection.add(
                    ids=[str(i)],
                    documents=[content],
                    metadatas=[metadata]
                )
            st.success(f"Loaded {len(data['feed'])} news items into ChromaDB.")
        else:
            st.error("No news data found.")
    except Exception as e:
        st.error(f"Error loading news data: {e}")

### Function to Load Ticker Trends Data into ChromaDB ###
def load_ticker_trends_data():
    try:
        response = requests.get(tickers_url)
        response.raise_for_status()
        data = response.json()

        if "top_gainers" in data and "top_losers" in data:
            ticker_collection.add(
                ids=["top_gainers"],
                documents=[json.dumps(data["top_gainers"])],
                metadatas=[{"type": "Top Gainers"}]
            )
            ticker_collection.add(
                ids=["top_losers"],
                documents=[json.dumps(data["top_losers"])],
                metadatas=[{"type": "Top Losers"}]
            )
            st.success("Ticker trends data loaded into ChromaDB.")
        else:
            st.error("Invalid data format from API.")
    except Exception as e:
        st.error(f"Error loading ticker trends data: {e}")

### Function to Query Data from ChromaDB ###
def query_chromadb(collection, query_text):
    try:
        results = collection.query(
            query_text=query_text,
            n_results=5
        )
        return results["documents"], results["metadatas"]
    except Exception as e:
        st.error(f"Error querying data: {e}")
        return [], []

### Function to Generate Newsletter ###
def generate_newsletter():
    try:
        # Query News Data
        news_query = "Latest financial news"
        news_docs, news_metadata = query_chromadb(news_collection, news_query)

        # Query Ticker Trends Data
        ticker_query = "Top market gainers and losers"
        ticker_docs, ticker_metadata = query_chromadb(ticker_collection, ticker_query)

        # Combine Data for OpenAI
        combined_context = f"News Data: {news_docs}\nTicker Data: {ticker_docs}"
        
        # Generate Newsletter using OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial newsletter editor."},
                {"role": "user", "content": f"Generate a newsletter using the following data:\n{combined_context}"}
            ]
        )
        newsletter = response["choices"][0]["message"]["content"]

        # Display the Newsletter
        st.subheader("Generated Newsletter")
        st.write(newsletter)
    except Exception as e:
        st.error(f"Error generating newsletter: {e}")

# Main Logic
if option == "Load News Data":
    load_news_data()
elif option == "Retrieve News Data":
    query = st.text_input("Enter your query for news data:")
    if st.button("Search News"):
        docs, metadatas = query_chromadb(news_collection, query)
        for doc, meta in zip(docs, metadatas):
            st.write(f"Title: {meta['title']}")
            st.write(f"Source: {meta['source']}")
            st.write(f"Content: {doc}")
            st.write("---")
elif option == "Load Ticker Trends Data":
    load_ticker_trends_data()
elif option == "Retrieve Ticker Trends Data":
    query = st.text_input("Enter your query for ticker trends:")
    if st.button("Search Ticker Trends"):
        docs, metadatas = query_chromadb(ticker_collection, query)
        for doc, meta in zip(docs, metadatas):
            st.write(f"Type: {meta['type']}")
            st.write(f"Content: {doc}")
            st.write("---")
elif option == "Generate Newsletter":
    generate_newsletter()
