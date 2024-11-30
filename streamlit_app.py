import streamlit as st
import requests
import json
import openai
from crewai import Crew, Process, Agent, Task
from langchain.chat_models import ChatOpenAI

# Import pysqlite3 for chromadb compatibility
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai.api_key = st.secrets["openai"]["api_key"]

# Initialize ChatOpenAI Model
openai_llm = ChatOpenAI(model="gpt-4", openai_api_key=openai.api_key)

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent System with CrewAI and Tasks")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:",
    ["Load News Data", "Retrieve News Data", "Load Ticker Trends Data", "Retrieve Ticker Trends Data", "Generate Newsletter"]
)

### Function Definitions for Data Loading and Retrieval ###

def load_news_data():
    news_collection = client.get_or_create_collection("news_sentiment_data")
    existing_data = news_collection.get()
    if existing_data["documents"]:
        st.warning("News data already exists in ChromaDB. No API call made.")
        return
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
    except Exception as e:
        st.error(f"Error: {e}")

def retrieve_news_data():
    news_collection = client.get_or_create_collection("news_sentiment_data")
    doc_id = st.text_input("Enter the News Document ID to retrieve:", "1")
    if st.button("Retrieve News"):
        try:
            results = news_collection.get(ids=[doc_id])
            if results['documents']:
                for document in results['documents']:
                    parsed_document = json.loads(document)
                    st.write("### News Data")
                    st.json(parsed_document)
            else:
                st.warning("No data found.")
        except Exception as e:
            st.error(f"Error: {e}")

def load_ticker_trends_data():
    ticker_collection = client.get_or_create_collection("ticker_trends_data")
    existing_data = ticker_collection.get()
    if existing_data["documents"]:
        st.warning("Ticker trends data already exists in ChromaDB. No API call made.")
        return
    try:
        response = requests.get(tickers_url)
        response.raise_for_status()
        data = response.json()
        if "top_gainers" in data:
            ticker_collection.add(
                ids=["top_gainers"],
                metadatas=[{"type": "top_gainers"}],
                documents=[json.dumps(data["top_gainers"])],
            )
            ticker_collection.add(
                ids=["top_losers"],
                metadatas=[{"type": "top_losers"}],
                documents=[json.dumps(data["top_losers"])],
            )
            ticker_collection.add(
                ids=["most_actively_traded"],
                metadatas=[{"type": "most_actively_traded"}],
                documents=[json.dumps(data["most_actively_traded"])],
            )
            st.success("Ticker trends data added to ChromaDB.")
        else:
            st.error("Invalid data format received from API.")
    except Exception as e:
        st.error(f"Error: {e}")

def retrieve_ticker_trends_data():
    ticker_collection = client.get_or_create_collection("ticker_trends_data")
    data_type = st.radio("Data Type", ["Top Gainers", "Top Losers", "Most Actively Traded"])
    if st.button("Retrieve Data"):
        try:
            results = ticker_collection.get(ids=[data_type.lower().replace(" ", "_")])
            if results["documents"]:
                st.json(json.loads(results["documents"][0]))
            else:
                st.warning("No data found.")
        except Exception as e:
            st.error(f"Error: {e}")

### Crew, Agents, and Tasks ###

researcher = Agent(role="Researcher", goal="Process news data", backstory="Experienced researcher.", llm=openai_llm)
market_analyst = Agent(role="Market Analyst", goal="Analyze trends", backstory="Market trends expert.", llm=openai_llm)
risk_analyst = Agent(role="Risk Analyst", goal="Identify risks", backstory="Experienced in risk analysis.", llm=openai_llm)
writer = Agent(role="Writer", goal="Generate newsletter", backstory="Expert in content creation.", llm=openai_llm)

news_task = Task(description="Extract insights from news data", agent=researcher)
market_trends_task = Task(description="Analyze market trends", agent=market_analyst)
risk_analysis_task = Task(description="Analyze risk data", agent=risk_analyst)
newsletter_task = Task(description="Write the newsletter", agent=writer)

report_crew = Crew(
    agents=[researcher, market_analyst, risk_analyst, writer],
    tasks=[news_task, market_trends_task, risk_analysis_task, newsletter_task],
    process=Process.sequential
)

def generate_newsletter_with_tasks():
    result = report_crew.kickoff()
    newsletter = result.tasks[-1].output  # Get the final task's output
    st.subheader("Generated Newsletter")
    st.text(newsletter)

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
    generate_newsletter_with_tasks()
