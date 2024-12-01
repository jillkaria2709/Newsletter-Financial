import streamlit as st
import requests
import json
import openai
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from bespokelabs import BespokeLabs
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

# Initialize Bespoke Labs with the API key
bl = BespokeLabs(auth_token=st.secrets["bespoke"]["api_key"])

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai.api_key = st.secrets["openai"]["api_key"]

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent System with RAG, Bespoke Labs, and Chatbot")

### Helper Functions ###
def scrape_context_from_url(url):
    """Scrape relevant context from the given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=" ").strip()
        return text[:2000]  # Limit the scraped text to 2000 characters for Bespoke
    except Exception as e:
        st.error(f"Error scraping context from URL: {e}")
        return None

def factcheck_with_bespoke_from_newsletter(newsletter):
    """Perform fact-checking using Bespoke Labs with dynamically retrieved context."""
    if not newsletter:
        st.error("No newsletter content available for fact-checking.")
        return None

    # Use the newsletter content as the claim
    claim = newsletter

    # Dynamically retrieve context from multiple sources
    context_sources = [
        "https://www.alphavantage.co",  # Example financial API site
        "https://www.bespokepremium.com",  # Bespoke Premium site
        "https://www.marketwatch.com",  # MarketWatch for financial news
    ]

    context = ""
    for url in context_sources:
        scraped_context = scrape_context_from_url(url)
        if scraped_context:
            context += f"\n\nContext from {url}:\n{scraped_context}"

    if context:
        try:
            response = bl.minicheck.factcheck.create(claim=claim, context=context)
            return {
                "support_prob": getattr(response, "support_prob", "N/A"),
                "details": str(response)
            }
        except Exception as e:
            st.error(f"Error with Bespoke Labs Fact-Check: {e}")
            return None
    else:
        st.error("Failed to retrieve context for fact-checking.")
        return None

def call_openai_gpt4(prompt):
    """Call OpenAI GPT-4 to process the prompt."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
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
        # Placeholder for data retrieval (e.g., ChromaDB)
        retrieved_data = []

        combined_data = retrieved_data
        if additional_data:
            combined_data.extend(additional_data)

        augmented_prompt = (
            f"Role: {self.role}\nGoal: {self.goal}\n"
            f"Task: {task_description}\nRelevant Data:\n{json.dumps(combined_data)}\n"
            "Please provide a detailed response using financial jargon. Keep it concise yet comprehensive."
        )
        result = call_openai_gpt4(augmented_prompt)
        return result

### Agents and Tasks ###
researcher = RAGAgent(role="Researcher", goal="Process news data")
market_analyst = RAGAgent(role="Market Analyst", goal="Analyze trends")
risk_analyst = RAGAgent(role="Risk Analyst", goal="Identify risks")
writer = RAGAgent(role="Writer", goal="Generate newsletter")

### Newsletter Generation ###
def generate_sequential_newsletter(news_insights, market_insights, risk_insights):
    """Generate a professional newsletter using financial jargon and detailed agent outputs."""
    if news_insights and market_insights and risk_insights:
        combined_context = f"""
        Researcher Insights: {news_insights}
        Market Analyst Insights: {market_insights}
        Risk Analyst Insights: {risk_insights}
        """

        writer_task_description = (
            "Write a comprehensive newsletter based on the following insights. "
            "The newsletter should target an audience familiar with financial markets and economic trends, "
            "using appropriate financial jargon and terminologies. Focus on key metrics, trends, risks, and "
            "actionable insights, ensuring the output is concise, informative, and approximately 2,000 tokens in length."
        )
        newsletter = writer.execute_task(writer_task_description, additional_data=[combined_context])

        if "Error" in newsletter:
            st.error("Failed to generate the newsletter.")
        else:
            st.session_state["newsletter_content"] = newsletter  # Store newsletter for fact-checking
            st.subheader("Generated Newsletter")
            st.markdown(f"## Writer's Newsletter\n{newsletter}")
    else:
        st.error("Missing insights for generating the newsletter.")

### Main Buttons ###
if st.button("Fetch and Store News Data"):
    st.write("Fetching and storing news data...")
    # Code for fetching and storing news data goes here

if st.button("Fetch and Store Trends Data"):
    st.write("Fetching and storing trends data...")
    # Code for fetching and storing trends data goes here

if st.button("Generate Newsletter"):
    st.write("Generating newsletter...")
    news_insights = researcher.execute_task("Extract insights from news data")
    market_insights = market_analyst.execute_task("Analyze market trends")
    risk_insights = risk_analyst.execute_task("Identify risks", additional_data=[news_insights, market_insights])
    generate_sequential_newsletter(news_insights, market_insights, risk_insights)

### Fact-Checking Integration ###
st.subheader("Fact-Check with Bespoke Labs")

if st.button("Fact-Check Newsletter"):
    try:
        # Use the last generated newsletter as the claim
        newsletter_content = st.session_state.get("newsletter_content", None)
        if not newsletter_content:
            st.error("No newsletter content found. Please generate a newsletter first.")
        else:
            factcheck_result = factcheck_with_bespoke_from_newsletter(newsletter_content)
            if factcheck_result:
                st.write("**Fact-Check Result**")
                st.write(f"Support Probability: {factcheck_result['support_prob']}")
                st.write(f"Details: {factcheck_result['details']}")
    except Exception as e:
        st.error(f"Error during fact-checking: {e}")
