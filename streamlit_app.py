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

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai.api_key = st.secrets["openai"]["api_key"]

# Initialize Bespoke Labs
bl = BespokeLabs(
    auth_token=st.secrets["bespoke_labs"]["api_key"]
)

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

        # Fact-checking with Bespoke
        try:
            factcheck_response = bl.minicheck.factcheck.create(
                claim=newsletter,
                context=input_text
            )
            # Display fact-checking results
            if hasattr(factcheck_response, "support_prob"):
                support_prob = factcheck_response.support_prob
                st.write(f"Support Probability: {support_prob:.2f}")
                if support_prob > 0.75:
                    st.success("The newsletter is likely accurate.")
                elif support_prob > 0.5:
                    st.warning("The newsletter is somewhat supported, but additional review is recommended.")
                else:
                    st.error("The newsletter may not be accurate. Consider revising the content.")
            else:
                st.error("Fact-checking response does not include a support probability.")
                st.json(factcheck_response)  # Log raw response for debugging
        except Exception as e:
            st.error(f"Error in Bespoke fact-checking: {e}")

    except Exception as e:
        st.error(f"Error generating newsletter and fact-checking: {e}")

# Main Logic
if option == "Load News Data":
    load_news_data()
elif option == "Load Ticker Trends Data":
    load_ticker_trends_data()
elif option == "Generate Newsletter & Fact-Check":
    generate_newsletter_and_factcheck()
