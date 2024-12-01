import streamlit as st
import requests
import json
import openai
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from bespokelabs import BespokeLabs

# Initialize Bespoke Labs
bl = BespokeLabs(
    auth_token=st.secrets["bespoke"]["api_key"],
)

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai.api_key = st.secrets["openai"]["api_key"]

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent System with RAG and OpenAI GPT-4")

### Helper Functions ###

def retrieve_top_news(collection_name, query, top_k=3):
    """Retrieve top K most relevant news articles based on relevance_score_definition."""
    try:
        collection = client.get_or_create_collection(collection_name)
        results = collection.query(
            query_texts=[query],
            n_results=10
        )
        # Process and rank results based on relevance_score_definition
        parsed_docs = [
            json.loads(doc) if isinstance(doc, str) else doc
            for doc in results.get('documents', [])
        ]
        ranked_articles = sorted(
            parsed_docs,
            key=lambda x: x.get("relevance_score_definition", 0),
            reverse=True
        )[:top_k]
        return ranked_articles
    except Exception as e:
        st.error(f"Error retrieving data from {collection_name}: {e}")
        return []

def update_chromadb(collection_name, data):
    """Update ChromaDB with new data."""
    collection = client.get_or_create_collection(collection_name)
    for i, item in enumerate(data, start=1):
        collection.add(
            ids=[str(i)],
            metadatas=[{"source": item.get("source", "N/A"), "time_published": item.get("time_published", "N/A")}],
            documents=[json.dumps(item)]
        )

def fetch_and_update_news_data():
    """Fetch news data from the API and update ChromaDB."""
    try:
        response = requests.get(news_url)
        response.raise_for_status()
        data = response.json()
        st.write("News API Response:", data)  # Print the API response
        if 'feed' in data:
            update_chromadb("news_sentiment_data", data['feed'])
            st.success("News data updated in ChromaDB.")
        else:
            st.error("No news data found in API response.")
    except Exception as e:
        st.error(f"Error updating news data: {e}")

def fetch_and_update_ticker_trends_data():
    """Fetch ticker trends data from the API and update ChromaDB."""
    try:
        response = requests.get(tickers_url)
        response.raise_for_status()
        data = response.json()
        st.write("Ticker Trends API Response:", data)  # Print the API response
        if "top_gainers" in data:
            combined_data = [
                {"type": "top_gainers", "data": data["top_gainers"]},
                {"type": "top_losers", "data": data["top_losers"]},
                {"type": "most_actively_traded", "data": data["most_actively_traded"]}
            ]
            update_chromadb("ticker_trends_data", combined_data)
            st.success("Ticker trends data updated in ChromaDB.")
        else:
            st.error("Invalid data format received from API.")
    except Exception as e:
        st.error(f"Error updating ticker trends data: {e}")

def call_openai_gpt4(prompt):
    """Call OpenAI GPT-4 to process the prompt."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error calling OpenAI GPT-4: {e}")
        return "I'm sorry, I couldn't process your request at this time."

### Agents ###

class RAGAgent:
    def __init__(self, role, goal):
        self.role = role
        self.goal = goal

    def execute_task(self, task_description, additional_data=None):
        """Execute the task using RAG and OpenAI GPT-4 for formatting."""
        combined_data = additional_data or []
        formatted_data = json.dumps(combined_data, indent=2)

        prompt = (
            f"Role: {self.role}\nGoal: {self.goal}\nTask: {task_description}\n\n"
            f"Use ONLY the following RAG data to frame your response:\n{formatted_data}"
        )

        return call_openai_gpt4(prompt)

### Newsletter Generation ###

def generate_newsletter_with_rag():
    """Generate the newsletter using RAG and agents and validate it with Bespoke Labs."""
    st.write("Executing: Extract insights from news data (Researcher)")
    top_news = retrieve_top_news("news_sentiment_data", "Market trends", top_k=3)
    news_results = researcher.execute_task("Extract insights from the top 3 news articles.", additional_data=top_news)

    st.write("Executing: Analyze market trends (Market Analyst)")
    trends_results = market_analyst.execute_task("Analyze market trends based on the top 3 news articles.", additional_data=top_news)

    st.write("Executing: Analyze risk data (Risk Analyst)")
    risk_context = [{"source": "news_sentiment_data", "content": top_news}]
    risk_results = risk_analyst.execute_task("Identify risks in the current market landscape.", additional_data=risk_context)

    context = {
        "RAG_Data": {"News": top_news},
        "Agent_Insights": {
            "Researcher": news_results,
            "Market Analyst": trends_results,
            "Risk Analyst": risk_results
        }
    }

    st.write("Executing: Write the newsletter (Writer)")
    newsletter = writer.execute_task("Generate a market insights newsletter.", additional_data=context)

    st.subheader("Generated Newsletter")
    st.markdown(newsletter)

    try:
        st.write("Validating the newsletter with Bespoke Labs...")
        factcheck_response = bl.minicheck.factcheck.create(
            claim=newsletter,
            context=json.dumps(context)
        )
        support_prob = factcheck_response.get("support_prob", None)
        if support_prob:
            st.write(f"Fact-Check Support Probability: {support_prob}")
        else:
            st.error("No support probability returned.")
    except Exception as e:
        st.error(f"Error during newsletter validation: {e}")

### Main Page Buttons ###

researcher = RAGAgent(role="Researcher", goal="Process news data")
market_analyst = RAGAgent(role="Market Analyst", goal="Analyze trends")
risk_analyst = RAGAgent(role="Risk Analyst", goal="Identify risks")
writer = RAGAgent(role="Writer", goal="Generate newsletter")

if st.button("Fetch and Store News Data"):
    fetch_and_update_news_data()

if st.button("Fetch and Store Trends Data"):
    fetch_and_update_ticker_trends_data()

if st.button("Generate Newsletter"):
    generate_newsletter_with_rag()
