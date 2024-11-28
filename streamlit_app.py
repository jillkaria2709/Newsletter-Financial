import streamlit as st
import requests
import json
__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings
from crewai import Agent
from crewai_tools.tools.rag_tool import RagTool
from openai import OpenAI

# Initialize OpenAI and ChromaDB Clients
openai_client = OpenAI(api_key=st.secrets["api_keys"]["openai"])
client = chromadb.PersistentClient()

# API key and URLs for Alpha Vantage
api_key = st.secrets["api_keys"]["alpha_vantage"]
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

# Initialize RagTool
news_rag_tool = RagTool()
ticker_rag_tool = RagTool()

### Function to Load News Data into RagTool ###
def load_news_data():
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
                # Add document to RagTool
                news_rag_tool.add_document(content=json.dumps(document), metadata={"source": "news", "id": str(i)})
            st.success(f"Loaded {len(news_items)} news items into RagTool.")
        else:
            st.error("No news data found.")
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

### Function to Load Ticker Trends Data into RagTool ###
def load_ticker_trends_data():
    try:
        response = requests.get(tickers_url)
        response.raise_for_status()
        data = response.json()

        if "metadata" in data and "top_gainers" in data:
            metadata = {"metadata": data["metadata"], "last_updated": data["last_updated"]}
            top_gainers = data["top_gainers"]
            top_losers = data["top_losers"]
            most_actively_traded = data["most_actively_traded"]

            # Add documents to RagTool
            ticker_rag_tool.add_document(content=json.dumps(top_gainers), metadata={"type": "top_gainers"})
            ticker_rag_tool.add_document(content=json.dumps(top_losers), metadata={"type": "top_losers"})
            ticker_rag_tool.add_document(content=json.dumps(most_actively_traded), metadata={"type": "most_actively_traded"})

            st.success("Ticker trends data added to RagTool.")
        else:
            st.error("Invalid data format received from API.")
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

### Function to Retrieve Data from RagTool ###
def retrieve_data(rag_tool, query):
    try:
        results = rag_tool.query({"query_text": query, "n_results": 5})
        for result in results:
            st.json(result)
    except Exception as e:
        st.error(f"Error retrieving data: {e}")

### Function to Generate Newsletter ###
def generate_newsletter():
    try:
        news_query = "What are the latest financial news trends?"
        ticker_query = "What are the top market gainers and losers?"

        # Use RagTool to query data
        news_data = news_rag_tool.query({"query_text": news_query, "n_results": 5})
        ticker_data = ticker_rag_tool.query({"query_text": ticker_query, "n_results": 5})

        # Combine results and send to OpenAI
        combined_context = f"News Data: {news_data}\nTicker Data: {ticker_data}"
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial newsletter editor."},
                {"role": "user", "content": f"Generate a newsletter based on this context:\n{combined_context}"}
            ]
        )
        newsletter = response["choices"][0]["message"]["content"]
        st.subheader("Generated Newsletter")
        st.write(newsletter)
    except Exception as e:
        st.error(f"Error generating newsletter: {e}")

# Main Logic
if option == "Load News Data":
    load_news_data()
elif option == "Retrieve News Data":
    query = st.text_input("Enter query for news data:")
    if st.button("Search News"):
        retrieve_data(news_rag_tool, query)
elif option == "Load Ticker Trends Data":
    load_ticker_trends_data()
elif option == "Retrieve Ticker Trends Data":
    query = st.text_input("Enter query for ticker trends:")
    if st.button("Search Ticker Trends"):
        retrieve_data(ticker_rag_tool, query)
elif option == "Generate Newsletter":
    generate_newsletter()
