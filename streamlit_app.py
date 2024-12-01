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

news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent System with RAG, GPT-4, and Bespoke Labs")

### Helper Functions ###

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
        if 'feed' in data:
            update_chromadb("news_sentiment_data", data['feed'])
            st.success("News data updated in ChromaDB.")
        else:
            st.error("No news data found in API response.")
    except Exception as e:
        st.error(f"Error updating news data: {e}")

def retrieve_top_news_articles(collection_name, top_k=3):
    """Retrieve top K news articles based on relevance_score_definition."""
    try:
        # Access the collection
        collection = client.get_or_create_collection(collection_name)
        
        # Fetch all documents with a broad query
        results = collection.query(
            query_texts=[""],  # Broad query to fetch documents
            n_results=100  # Fetch a large number of results for ranking
        )
        
        # Directly handle the case where results are a list
        if isinstance(results, dict) and 'documents' in results:
            articles = [
                json.loads(doc) if isinstance(doc, str) else doc
                for doc in results['documents']
            ]
        elif isinstance(results, list):
            articles = [
                json.loads(doc) if isinstance(doc, str) else doc
                for doc in results
            ]
        else:
            articles = []
        
        # Ensure each article is a dictionary
        parsed_articles = [
            article if isinstance(article, dict) else json.loads(article)
            for article in articles
        ]
        
        # Sort articles by `relevance_score_definition`
        sorted_articles = sorted(
            parsed_articles,
            key=lambda x: x.get("relevance_score_definition", 0),
            reverse=True
        )
        
        # Return the top K articles
        return sorted_articles[:top_k]
    except Exception as e:
        st.error(f"Error retrieving top articles from {collection_name}: {e}")
        return []

def call_openai_gpt4(prompt):
    """Call OpenAI GPT-4 to process the prompt."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        return content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI GPT-4: {e}")
        return "I'm sorry, I couldn't process your request at this time."

class RAGAgent:
    def __init__(self, role, goal):
        self.role = role
        self.goal = goal

    def execute_task(self, task_description, additional_data=None):
        """Execute the task using RAG and GPT-4 summarization."""
        combined_data = additional_data or []
        augmented_prompt = f"Role: {self.role}\nGoal: {self.goal}\nTask: {task_description}\nRelevant Data:\n{json.dumps(combined_data)}"
        return call_openai_gpt4(augmented_prompt)

### Agents Initialization ###
researcher = RAGAgent(role="Researcher", goal="Process news data")
market_analyst = RAGAgent(role="Market Analyst", goal="Analyze trends")
risk_analyst = RAGAgent(role="Risk Analyst", goal="Identify risks")
writer = RAGAgent(role="Writer", goal="Generate newsletter")

### Newsletter Generation ###
def generate_newsletter_with_rag():
    """Generate the newsletter using RAG and agents."""
    st.write("Executing: Extract insights from news data (Researcher)")
    top_news = retrieve_top_news_articles("news_sentiment_data", top_k=3)
    news_results = researcher.execute_task("Extract insights from news data", additional_data=top_news)

    st.write("Executing: Analyze market trends (Market Analyst)")
    trends_results = market_analyst.execute_task("Analyze market trends", additional_data=top_news)

    st.write("Executing: Analyze risk data (Risk Analyst)")
    risk_context = [{"source": "news_sentiment_data", "content": top_news}]
    risk_results = risk_analyst.execute_task("Analyze risk data", additional_data=risk_context)

    context = {
        "RAG_Data": {"News": top_news},
        "Agent_Insights": {
            "Researcher": news_results,
            "Market Analyst": trends_results,
            "Risk Analyst": risk_results
        }
    }

    st.write("Executing: Write the newsletter (Writer)")
    writer_task_description = "Write a cohesive newsletter based on insights from news, market trends, and risk analysis."
    newsletter = writer.execute_task(writer_task_description, additional_data=context)

    st.subheader("Generated Newsletter")
    st.markdown(newsletter)

    # Validate with Bespoke Labs
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
if st.button("Fetch and Store News Data"):
    fetch_and_update_news_data()

if st.button("Generate Newsletter"):
    generate_newsletter_with_rag()
