import streamlit as st
import requests
import json
import chromadb
from chromadb.config import Settings
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
st.title("Alpha Vantage RAG System")

# Sidebar options
st.sidebar.header("Options")
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
            for i, item in enumerate(data['feed'], start=1):
                document = {
                    "id": str(i),
                    "title": item["title"],
                    "url": item["url"],
                    "time_published": item["time_published"],
                    "source": item["source"],
                    "summary": item.get("summary", "N/A"),
                    "topics": [topic["topic"] for topic in item.get("topics", [])],
                    "overall_sentiment_label": item.get("overall_sentiment_label", "N/A"),
                    "overall_sentiment_score": item.get("overall_sentiment_score", "N/A")
                }

                # Add to ChromaDB
                news_collection.add(
                    ids=[document["id"]],
                    metadatas=[{
                        "title": document["title"],
                        "source": document["source"],
                        "time_published": document["time_published"],
                        "overall_sentiment": document["overall_sentiment_label"]
                    }],
                    documents=[json.dumps(document)]
                )

            st.success(f"Loaded {len(data['feed'])} news items into ChromaDB.")
        else:
            st.error("No news data found.")
    except Exception as e:
        st.error(f"Error loading news data: {e}")

### Function to Retrieve News Data from ChromaDB ###
def retrieve_news_data():
    doc_id = st.text_input("Enter the News Document ID to retrieve:", "1")

    if st.button("Retrieve News"):
        try:
            results = news_collection.get(ids=[doc_id])
            if results['documents']:
                for document, metadata in zip(results['documents'], results['metadatas']):
                    parsed_document = json.loads(document)
                    st.write("### News Data")
                    st.json(parsed_document)
            else:
                st.warning("No data found for the given News ID.")
        except Exception as e:
            st.error(f"Error retrieving news data: {e}")

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

### Function to Retrieve Ticker Trends Data ###
def retrieve_ticker_trends_data():
    data_type = st.radio(
        "Data Type", ["Top Gainers", "Top Losers"]
    )

    data_id_mapping = {
        "Top Gainers": "top_gainers",
        "Top Losers": "top_losers"
    }

    if st.button("Retrieve Ticker Trends Data"):
        try:
            results = ticker_collection.get(ids=[data_id_mapping[data_type]])
            if results["documents"]:
                for document, metadata in zip(results["documents"], results["metadatas"]):
                    st.write(f"### {metadata['type']}")
                    st.json(json.loads(document))
            else:
                st.warning("No data found.")
        except Exception as e:
            st.error(f"Error retrieving ticker trends data: {e}")

def generate_newsletter():
    try:
        # Retrieve all documents from the news collection
        news_results = news_collection.query(
            query_texts=[""],  # Empty query text to fetch all documents
            n_results=1000     # A high number to ensure all documents are fetched
        )
        news_content = "\n".join(
            [f"{meta['title']}: {json.loads(doc)['summary']}" for doc, meta in zip(news_results["documents"], news_results["metadatas"])]
        )

        # Retrieve all documents from the ticker trends collection
        ticker_results = ticker_collection.query(
            query_texts=[""],  # Empty query text to fetch all documents
            n_results=1000     # A high number to ensure all documents are fetched
        )
        ticker_content = "\n".join(
            [f"{meta['type']}: {doc}" for doc, meta in zip(ticker_results["documents"], ticker_results["metadatas"])]
        )

        # Combine data for OpenAI
        combined_data = f"News Summaries:\n{news_content}\n\nTicker Trends:\n{ticker_content}"

        # Generate newsletter using OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial newsletter editor."},
                {"role": "user", "content": f"Generate a newsletter using the following data:\n{combined_data}"}
            ]
        )
        newsletter = response["choices"][0]["message"]["content"]

        # Display the newsletter
        st.subheader("Generated Newsletter")
        st.write(newsletter)
    except Exception as e:
        st.error(f"Error generating newsletter: {e}")


# Main Logic
if option == "Load News Data":
    load_news_data()
elif option == "Retrieve News Data":
    retrieve_news_data()
elif option == "Load Ticker Trends Data":
    load_ticker_trends_data()
elif option == "Retrieve Ticker Trends Data":
    retrieve_ticker_trends_data()
elif option == "Generate Newsletter":
    generate_newsletter()
