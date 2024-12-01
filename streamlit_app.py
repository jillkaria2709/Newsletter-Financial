import streamlit as st
import requests
import json
import openai
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from bespokelabs import BespokeLabs

# Initialize Bespoke Labs with the API key
bl = BespokeLabs(auth_token=st.secrets["bespoke"]["api_key"])

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai.api_key = st.secrets["openai"]["api_key"]

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent System with RAG, Bespoke Labs, Chatbot, and More")

### Define Function Schema ###
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_daily_data",
            "description": "Fetch daily data for a specific type (e.g., stock prices, news, or trends).",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_type": {
                        "type": "string",
                        "enum": ["stocks", "news", "trends"],
                        "description": "The type of daily data to fetch (e.g., stocks, news, or trends)."
                    },
                    "symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol (if data_type is 'stocks')."
                    }
                },
                "required": ["data_type"]
            }
        }
    }
]

### Helper Functions ###
def fetch_ticker_price(ticker):
    """Fetch the latest price for a given ticker symbol."""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={alpha_vantage_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "Time Series (Daily)" in data:
            latest_date = max(data["Time Series (Daily)"].keys())
            latest_prices = data["Time Series (Daily)"][latest_date]
            return {
                "ticker": ticker,
                "date": latest_date,
                "open": latest_prices["1. open"],
                "high": latest_prices["2. high"],
                "low": latest_prices["3. low"],
                "close": latest_prices["4. close"],
                "volume": latest_prices["5. volume"]
            }
        else:
            return {"error": "Ticker symbol not found or invalid data received."}
    except Exception as e:
        return {"error": f"Error fetching ticker data: {e}"}

def get_daily_data(data_type, symbol=None):
    """Fetch daily data based on the specified type."""
    try:
        if data_type == "stocks":
            if not symbol:
                return {"error": "Symbol is required for stock data."}
            return fetch_ticker_price(symbol)

        elif data_type == "news":
            news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=5"
            response = requests.get(news_url)
            response.raise_for_status()
            news_data = response.json()
            return news_data.get("feed", [])

        elif data_type == "trends":
            trends_url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}"
            response = requests.get(trends_url)
            response.raise_for_status()
            trends_data = response.json()
            return trends_data
        else:
            return {"error": "Invalid data type. Choose from 'stocks', 'news', or 'trends'."}
    except Exception as e:
        return {"error": str(e)}

def retrieve_from_chromadb(collection_name, query, top_k=3):
    """Retrieve relevant documents from ChromaDB."""
    collection = client.get_or_create_collection(collection_name)
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results['documents']
    except Exception as e:
        st.error(f"Error retrieving data from ChromaDB: {e}")
        return []

def call_openai_gpt4(prompt):
    """Call OpenAI GPT-4 to process the prompt."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        return content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI GPT-4: {e}")
        return "I'm sorry, I couldn't process your request at this time."

### Multi-Agent System ###
class RAGAgent:
    def __init__(self, role, goal):
        self.role = role
        self.goal = goal

    def execute_task(self, task_description, additional_data=None):
        """Execute a task based on the agent's role."""
        if "news" in self.goal.lower():
            retrieved_data = retrieve_from_chromadb("news_sentiment_data", task_description)
        elif "trends" in self.goal.lower():
            retrieved_data = retrieve_from_chromadb("market_data", task_description)
        else:
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

# Define agents
researcher = RAGAgent(role="Researcher", goal="Process news data")
market_analyst = RAGAgent(role="Market Analyst", goal="Analyze market trends")
risk_analyst = RAGAgent(role="Risk Analyst", goal="Identify risks")
writer = RAGAgent(role="Writer", goal="Generate newsletter")

### Newsletter Generation ###
def generate_sequential_newsletter(news_insights, market_insights, risk_insights):
    """Generate a professional newsletter using insights from agents."""
    if news_insights and market_insights and risk_insights:
        combined_context = f"""
        Researcher Insights: {news_insights}
        Market Analyst Insights: {market_insights}
        Risk Analyst Insights: {risk_insights}
        """

        writer_task_description = (
            "Role: Writer\n"
            "Goal: Generate a professional financial newsletter.\n"
            "Task: Write a comprehensive newsletter based on the provided insights. Ensure that:\n\n"
            "1. All statements are explicitly supported by the provided context.\n"
            "2. The newsletter uses financial jargon and concise, fact-based reporting.\n\n"
            f"Relevant Data:\n{combined_context}"
        )
        newsletter = writer.execute_task(writer_task_description, additional_data=[combined_context])
        st.subheader("Generated Newsletter")
        st.markdown(f"## Newsletter:\n\n{newsletter}")

### Buttons for Updating RAG Data and Newsletter ###
if st.button("Update News Data"):
    st.write("Updating news data...")
    # Fetch and update logic here

if st.button("Generate Newsletter"):
    st.write("Generating newsletter...")
    news_insights = researcher.execute_task("Analyze news data")
    market_insights = market_analyst.execute_task("Analyze market trends")
    risk_insights = risk_analyst.execute_task("Analyze risks")
    generate_sequential_newsletter(news_insights, market_insights, risk_insights)

if st.button("Fact-Check Newsletter"):
    st.write("Fact-checking the newsletter...")
    # Fact-checking logic here

### Chatbot ###
st.subheader("Chatbot")

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

user_input = st.text_input("Ask me something:")

if st.button("Send Query"):
    if len(user_input.strip()) == 0:
        st.write("Please enter a query.")
    else:
        user_input = user_input.strip()

        # Handle Ticker Queries
        if user_input.isalpha() and len(user_input) <= 5:
            ticker_result = fetch_ticker_price(user_input.upper())
            bot_response = json.dumps(ticker_result, indent=2)

        # Handle RAG-Based Questions
        elif "invest" in user_input.lower():
            retrieved_data = retrieve_from_chromadb("news_sentiment_data", user_input)
            bot_response = call_openai_gpt4(f"Context: {retrieved_data}\n\nQuestion: {user_input}")

        # Handle Out-of-Scope Questions
        else:
            bot_response = call_openai_gpt4(user_input)

        st.write(bot_response)
        st.session_state["conversation_history"].append({"user": user_input, "bot": bot_response})
