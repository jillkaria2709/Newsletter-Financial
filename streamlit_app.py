import streamlit as st
import requests
import json
import openai
import os
from bespokelabs import BespokeLabs
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_keys"]
openai.api_key = st.secrets["openai"]["api_keys"]
# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_keys"]
openai.api_key = st.secrets["openai"]["api_keys"]
bespoke_key = st.secrets["bespoke_labs"]["api_keys"]

# Initialize Bespoke Labs client
try:
    bl = BespokeLabs(auth_token=bespoke_key)
    st.success("Bespoke Labs client initialized successfully.")
except Exception as e:
    st.error(f"Failed to initialize Bespoke Labs client: {e}")
    bl = None


# API URLs
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage, Bespoke, & OpenAI RAG System")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:",
    ["Load News Data", "Load Ticker Trends Data", "Generate Newsletter & Fact-Check"]
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

### Function to Load Ticker Trends Data ###
def load_ticker_trends_data():
    ticker_collection = client.get_or_create_collection("ticker_trends_data")
    try:
        response = requests.get(tickers_url)
        response.raise_for_status()
        data = response.json()

        if "metadata" in data:
            # Store ticker trends data in ChromaDB
            ticker_collection.add(
                ids=["top_gainers"],
                metadatas=[{"type": "top_gainers"}],
                documents=[json.dumps(data.get("top_gainers", []))]
            )
            ticker_collection.add(
                ids=["top_losers"],
                metadatas=[{"type": "top_losers"}],
                documents=[json.dumps(data.get("top_losers", []))]
            )
            ticker_collection.add(
                ids=["most_actively_traded"],
                metadatas=[{"type": "most_actively_traded"}],
                documents=[json.dumps(data.get("most_actively_traded", []))]
            )
            st.success("Ticker trends data added to ChromaDB.")
        else:
            st.error("No ticker trends data found.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading ticker trends: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

### Function to Generate Newsletter & Fact-Check ###
def generate_newsletter_and_factcheck():
    news_collection = client.get_or_create_collection("news_sentiment_data")
    ticker_collection = client.get_or_create_collection("ticker_trends_data")
    
    try:
        # Retrieve all documents
        news_results = news_collection.get()
        news_data = [json.loads(doc) for doc in news_results["documents"]]

        # Retrieve ticker trends
        ticker_data = {}
        for data_type in ["top_gainers", "top_losers", "most_actively_traded"]:
            results = ticker_collection.get(ids=[data_type])
            ticker_data[data_type] = json.loads(results["documents"][0])

        # Combine data
        combined_data = {
            "news": news_data[:10],  # Limit news data for summarization
            "tickers": ticker_data
        }
        
        # Prepare data for summarization
        input_text = f"""
        News Data: {json.dumps(combined_data['news'], indent=2)}
        Ticker Trends:
        Top Gainers: {json.dumps(combined_data['tickers']['top_gainers'], indent=2)}
        Top Losers: {json.dumps(combined_data['tickers']['top_losers'], indent=2)}
        Most Actively Traded: {json.dumps(combined_data['tickers']['most_actively_traded'], indent=2)}
        """
        
        # Use OpenAI ChatCompletion API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant tasked with summarizing data into a concise newsletter."},
                {"role": "user", "content": f"Summarize the following RAG data into a concise newsletter:\n{input_text}"}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        # Extract the newsletter
        newsletter = response.choices[0].message.content.strip()
        
        # Display the newsletter
        st.subheader("Generated Newsletter")
        st.text(newsletter)

        # Fact-check the newsletter using Bespoke
        if not bl:
            st.error("Bespoke Labs client is not initialized. Skipping fact-checking.")
            return

        factcheck_response = bl.minicheck.factcheck.create(
            claim=newsletter,
            context=input_text
        )

        # Display fact-checking results
        support_prob = factcheck_response.support_prob
        st.write(f"Support Probability: {support_prob:.2f}")
        if support_prob > 0.75:
            st.success("The newsletter is likely accurate.")
        elif support_prob > 0.5:
            st.warning("The newsletter is somewhat supported, but additional review is recommended.")
        else:
            st.error("The newsletter may not be accurate. Consider revising the content.")
        
    except Exception as e:
        st.error(f"Error generating newsletter and fact-checking: {e}")

# Main Logic
if option == "Load News Data":
    load_news_data()
elif option == "Load Ticker Trends Data":
    load_ticker_trends_data()
elif option == "Generate Newsletter & Fact-Check":
    generate_newsletter_and_factcheck()
