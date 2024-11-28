import streamlit as st
import requests
import json
__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Replace 'YOUR_API_KEY' with your actual Alpha Vantage API key
api_key = 'H329FP3SD3PO0M7H'
url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={api_key}&limit=50'

# Streamlit App Title
st.title("News Sentiment RAG System")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio("Choose an action:", ["Load Data", "Retrieve Information"])

# Function to load data into ChromaDB
def load_data():
    # Create or access a collection in ChromaDB
    collection = client.get_or_create_collection("news_sentiment_data")

    # Fetch data from the API
    response = requests.get(url)
    data = response.json()

    if 'feed' in data:
        news_items = data['feed']
        for i, item in enumerate(news_items, start=1):
            # Prepare the document and metadata
            document = {
                "id": str(i),  # Unique identifier for each news item
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
            ticker_sentiments_str = json.dumps(document["ticker_sentiments"])  # Store as JSON string

            # Convert document to JSON string for ChromaDB
            document_str = json.dumps(document)

            # Insert the document into the ChromaDB collection
            collection.add(
                ids=[document["id"]],
                metadatas=[{
                    "source": document["source"],
                    "time_published": document["time_published"],
                    "topics": topics_str,  # Convert list to string
                    "overall_sentiment": document["overall_sentiment_label"],
                    "ticker_sentiments": ticker_sentiments_str,  # Store as JSON string
                }],
                documents=[document_str]  # Store full document
            )

        st.success(f"Inserted {len(news_items)} items into ChromaDB.")
    else:
        st.error("No news data found.")

# Function to retrieve information from ChromaDB
def retrieve_information():
    # Create or access the collection
    collection = client.get_or_create_collection("news_sentiment_data")
    
    # Input ID for retrieval
    doc_id = st.text_input("Enter the document ID to retrieve:", "1")

    if st.button("Retrieve"):
        try:
            results = collection.get(ids=[doc_id])
            if results['documents']:
                for document, metadata in zip(results['documents'], results['metadatas']):
                    parsed_document = json.loads(document)
                    parsed_ticker_sentiments = json.loads(metadata["ticker_sentiments"])
                    st.write("### News Title:", parsed_document["title"])
                    st.write("**Summary:**", parsed_document["summary"])
                    st.write("**Topics:**", metadata["topics"])
                    st.write("**Ticker Sentiments:**", parsed_ticker_sentiments)
            else:
                st.warning("No document found with the given ID.")
        except Exception as e:
            st.error(f"Error retrieving document: {e}")

# Main logic
if option == "Load Data":
    load_data()
elif option == "Retrieve Information":
    retrieve_information()
