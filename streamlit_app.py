import streamlit as st
import requests
import json
import chromadb
from chromadb.config import Settings
from crewai import Agent, Task, Crew, Process
from openai import ChatCompletion

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# API key and URLs for Alpha Vantage
api_key = st.secrets["api_keys"]["alpha_vantage"]
openai_api_key = st.secrets["api_keys"]["openai"]
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={api_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={api_key}'

# Streamlit App Title
st.title("Financial Insights Newsletter Generator")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:",
    ["Load News Data", "Retrieve News Data", "Load Ticker Trends Data", "Retrieve Ticker Trends Data", "Generate Newsletter"]
)

### Function to Load News Data into ChromaDB ###
def load_news_data():
    news_collection = client.get_or_create_collection("news_sentiment_data")
    try:
        response = requests.get(news_url)
        response.raise_for_status()
        data = response.json()
        if 'feed' in data:
            news_items = data['feed']
            for i, item in enumerate(news_items, start=1):
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
                topics_str = ", ".join(document["topics"])
                ticker_sentiments_str = json.dumps(document["ticker_sentiments"])
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
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

### Function to Retrieve News Data from ChromaDB ###
def retrieve_news_data():
    news_collection = client.get_or_create_collection("news_sentiment_data")
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
    ticker_collection = client.get_or_create_collection("ticker_trends_data")
    try:
        response = requests.get(tickers_url)
        response.raise_for_status()
        data = response.json()
        if "metadata" in data and "top_gainers" in data:
            metadata = {"metadata": data["metadata"], "last_updated": data["last_updated"]}
            top_gainers = data["top_gainers"]
            top_losers = data["top_losers"]
            most_actively_traded = data["most_actively_traded"]
            ticker_collection.add(
                ids=["metadata"],
                metadatas=[metadata],
                documents=["Ticker Trends Metadata"],
            )
            ticker_collection.add(
                ids=["top_gainers"],
                metadatas=[{"type": "top_gainers"}],
                documents=[json.dumps(top_gainers)],
            )
            ticker_collection.add(
                ids=["top_losers"],
                metadatas=[{"type": "top_losers"}],
                documents=[json.dumps(top_losers)],
            )
            ticker_collection.add(
                ids=["most_actively_traded"],
                metadatas=[{"type": "most_actively_traded"}],
                documents=[json.dumps(most_actively_traded)],
            )
            st.success("Ticker trends data added to ChromaDB.")
        else:
            st.error("Invalid data format received from API.")
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

### Function to Retrieve Ticker Trends Data ###
def retrieve_ticker_trends_data():
    ticker_collection = client.get_or_create_collection("ticker_trends_data")
    data_type = st.radio("Data Type", ["Metadata", "Top Gainers", "Top Losers", "Most Actively Traded"])
    data_id_mapping = {
        "Metadata": "metadata",
        "Top Gainers": "top_gainers",
        "Top Losers": "top_losers",
        "Most Actively Traded": "most_actively_traded",
    }
    if st.button("Retrieve Data"):
        try:
            results = ticker_collection.get(ids=[data_id_mapping[data_type]])
            if results["documents"]:
                for document, metadata in zip(results["documents"], results["metadatas"]):
                    st.write(f"### {data_type}")
                    if data_type == "Metadata":
                        st.json(metadata)
                    else:
                        st.json(json.loads(document))
            else:
                st.warning("No data found.")
        except Exception as e:
            st.error(f"Error retrieving data: {e}")

### CrewAI Agents ###
company_analyst_agent = Agent(
    role="Company Analyst",
    goal="Analyze news sentiment data to provide insights.",
    tools=[],
    llm=ChatCompletion(api_key=openai_api_key)
)

market_trends_agent = Agent(
    role="Market Trends Analyst",
    goal="Analyze ticker trends to provide market insights.",
    tools=[],
    llm=ChatCompletion(api_key=openai_api_key)
)

newsletter_agent = Agent(
    role="Newsletter Editor",
    goal="Compile a financial newsletter using all insights.",
    tools=[],
    llm=ChatCompletion(api_key=openai_api_key)
)

### Main Logic ###
if option == "Load News Data":
    load_news_data()
elif option == "Retrieve News Data":
    retrieve_news_data()
elif option == "Load Ticker Trends Data":
    load_ticker_trends_data()
elif option == "Retrieve Ticker Trends Data":
    retrieve_ticker_trends_data()
elif option == "Generate Newsletter":
    st.subheader("Generating Financial Newsletter...")
    st.write("Feature under development.")
