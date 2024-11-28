import streamlit as st
import requests
import json
import chromadb
from chromadb import PersistentClient
from crewai import Agent, Task, Crew, Process
import openai

# Custom ChromaDB Tool
class ChromaDBTool:
    def __init__(self, collection_name):
        self.client = PersistentClient()
        self.collection = self.client.get_or_create_collection(collection_name)
    
    def query(self, query_text, n_results=5):
        return self.collection.query(query_text=query_text, n_results=n_results)
    
    def add(self, ids, metadatas, documents):
        self.collection.add(ids=ids, metadatas=metadatas, documents=documents)

# Access API keys from Streamlit secrets
alpha_vantage_api_key = st.secrets["api_keys"]["alpha_vantage"]
openai_api_key = st.secrets["api_keys"]["openai"]

# Set OpenAI API key globally
openai.api_key = openai_api_key

# API URLs
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_api_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_api_key}'

# Streamlit App Title
st.title("Alpha Vantage RAG System")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:",
    ["Load News Data", "Retrieve News Data", "Load Ticker Trends Data", "Retrieve Ticker Trends Data", "Generate Newsletter"]
)

# Function to Load News Data into ChromaDB
def load_news_data():
    news_collection = PersistentClient().get_or_create_collection("news_sentiment_data")
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

# Function to Retrieve News Data from ChromaDB
def retrieve_news_data():
    news_collection = PersistentClient().get_or_create_collection("news_sentiment_data")
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

# Define CrewAI Agents
company_analyst_agent = Agent(
    role="Company Analyst",
    goal="Analyze news sentiment data to extract company-specific insights.",
    backstory="An experienced analyst in financial news.",
    tools=[ChromaDBTool(collection_name="news_sentiment_data")],
    llm=lambda prompt: openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )["choices"][0]["message"]["content"]
)

market_trends_agent = Agent(
    role="Market Trends Analyst",
    goal="Identify market trends from ticker data.",
    backstory="Specializes in market trends and financial performance.",
    tools=[ChromaDBTool(collection_name="ticker_trends_data")],
    llm=lambda prompt: openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )["choices"][0]["message"]["content"]
)

risk_management_agent = Agent(
    role="Risk Manager",
    goal="Assess market risks using insights from company analysis and market trends.",
    backstory="An expert in market risk assessment.",
    tools=[
        ChromaDBTool(collection_name="news_sentiment_data"),
        ChromaDBTool(collection_name="ticker_trends_data")
    ],
    llm=lambda prompt: openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )["choices"][0]["message"]["content"]
)

newsletter_agent = Agent(
    role="Newsletter Editor",
    goal="Compile insights into a well-formatted financial newsletter.",
    backstory="A skilled financial content creator.",
    llm=lambda prompt: openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )["choices"][0]["message"]["content"]
)

# Define CrewAI Tasks
company_analysis_task = Task(
    description="Analyze news sentiment data to extract insights on companies.",
    agent=company_analyst_agent
)

market_trends_task = Task(
    description="Analyze ticker data to identify market trends.",
    agent=market_trends_agent
)

risk_assessment_task = Task(
    description="Assess market risks based on company insights and market trends.",
    agent=risk_management_agent,
    context=[company_analysis_task, market_trends_task]
)

newsletter_compilation_task = Task(
    description="Compile all insights into a financial newsletter.",
    agent=newsletter_agent,
    context=[company_analysis_task, market_trends_task, risk_assessment_task]
)

# Assemble the Crew
financial_insights_crew = Crew(
    agents=[
        company_analyst_agent,
        market_trends_agent,
        risk_management_agent,
        newsletter_agent
    ],
    tasks=[
        company_analysis_task,
        market_trends_task,
        risk_assessment_task,
        newsletter_compilation_task
    ],
    process=Process.sequential,
    verbose=True
)

# Main Logic
if option == "Load News Data":
    load_news_data()
elif option == "Retrieve News Data":
    retrieve_news_data()
elif option == "Generate Newsletter":
    st.subheader("Generating Financial Newsletter")
    financial_insights_crew.kickoff()
    st.write("### Newsletter Output")
    st.text(newsletter_compilation_task.output)
