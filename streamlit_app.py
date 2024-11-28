import streamlit as st
import requests
import json
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# API key for Alpha Vantage
api_key = 'H329FP3SD3PO0M7H'

# API URLs
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={api_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={api_key}'

# Streamlit App Title
st.title("Alpha Vantage Data Manager")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:", 
    ["Load News Data", "Load Ticker Trends Data", "Retrieve Information"]
)

# Function to load news sentiment data into ChromaDB
def load_news_data():
    # Create or access a collection for news data
    news_collection = client.get_or_create_collection("news_sentiment_data")

    # Fetch data from the API
    response = requests.get(news_url)
    data = response.json()

    if 'feed' in data:
        news_items = data['feed']
        for i, item in enumerate(news_items, start=1):
            # Prepare the document and metadata
            document = {
                "id": str(i),
                "title": item["title"],
                "url": item["url"],
                "time_published": item["time_published"],
                "source": item["source"],
                "summary": item.get("summary", "N/A"),
                "topics": [topic["topic"] for topic in item.get("topics", [])],
                "overall_sentiment_label": item.get("overall_sentiment_label", "N/A"),
                "overall_sentiment_score": item.get("overall_sentiment_score", "N/A"),
                "ticker_sentiments": [
                    {
                        "ticker": ticker["ticker"],
                        "relevance_score": ticker["relevance_score"],
                        "ticker_sentiment_label": ticker["ticker_sentiment_label"],
                        "ticker_sentiment_score": ticker["ticker_sentiment_score"],
                    }
                    for ticker in item.get("ticker_sentiment", [])
                ],
            }
            
            # Convert lists in metadata to strings
            topics_str = ", ".join(document["topics"])
            ticker_sentiments_str = json.dumps(document["ticker_sentiments"])

            # Insert the document into the ChromaDB collection
            news_collection.add(
                ids=[document["id"]],
                metadatas=[{
                    "source": document["source"],
                    "time_published": document["time_published"],
                    "topics": topics_str,
                    "overall_sentiment": document["overall_sentiment_label"],
                    "ticker_sentiments": ticker_sentiments_str,
                }],
                documents=[json.dumps(document)]
            )

        st.success(f"Inserted {len(news_items)} news items into ChromaDB.")
    else:
        st.error("No news data found.")

# Function to load ticker trends data into ChromaDB
def load_ticker_trends_data():
    # Create or access a collection for ticker trends
    ticker_collection = client.get_or_create_collection("ticker_trends_data")

    # Fetch data from the API
    response = requests.get(tickers_url)
    data = response.json()

    # Validate data and store it
    if "metadata" in data and "top_gainers" in data:
        ticker_collection.add(
            ids=["ticker_trends_metadata"],
            metadatas=[{"type": "ticker_trends_metadata"}],
            documents=[json.dumps(data)]
        )
        st.success("Ticker trends data added to ChromaDB.")
    else:
        st.error("No ticker trends data found.")

# Function to retrieve information from ChromaDB
def retrieve_information():
    # Choose a collection
    collection_type = st.radio("Select Collection", ["News Sentiment", "Ticker Trends"])
    if collection_type == "News Sentiment":
        collection = client.get_or_create_collection("news_sentiment_data")
    else:
        collection = client.get_or_create_collection("ticker_trends_data")

    # Input ID for retrieval
    doc_id = st.text_input("Enter the document ID to retrieve:", "1")

    if st.button("Retrieve"):
        try:
            results = collection.get(ids=[doc_id])
            if results['documents']:
                for document, metadata in zip(results['documents'], results['metadatas']):
                    parsed_document = json.loads(document)
                    st.write("### Document Content")
                    st.json(parsed_document)
            else:
                st.warning("No document found with the given ID.")
        except Exception as e:
            st.error(f"Error retrieving document: {e}")

# Main logic
if option == "Load News Data":
    load_news_data()
elif option == "Load Ticker Trends Data":
    load_ticker_trends_data()
elif option == "Retrieve Information":
    retrieve_information()
