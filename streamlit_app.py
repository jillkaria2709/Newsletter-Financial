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

# API URLs for Alpha Vantage
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}'

# Streamlit App Title
st.title("Alpha Vantage Multi-Agent System with RAG, Bespoke Labs, and Chatbot")

### Helper Functions ###
def send_news_to_researcher():
    """Fetch 3 news items and send to Researcher agent."""
    response = requests.get(news_url)
    response.raise_for_status()
    data = response.json()

    if 'feed' in data:
        news_items = data['feed'][:3]  # Take only the first 3 news items
        news_insights = researcher.execute_task(
            task_description="Analyze the following news items.",
            additional_data=news_items
        )
        st.write("Researcher Insights:", news_insights)
        return news_insights
    else:
        st.error("No news data found in API response.")
        return None

### Step 2: Fetch Data and Send to Market Analyst ###

def send_market_data_to_analyst():
    """Fetch stock trends and send to Market Analyst agent."""
    response = requests.get(tickers_url)
    response.raise_for_status()
    data = response.json()

    if "top_gainers" in data and "top_losers" in data and "most_actively_traded" in data:
        market_data = [
            {"type": "top_gainers", "data": data["top_gainers"][:3]},  # Top 3 gainers
            {"type": "top_losers", "data": data["top_losers"][:3]},    # Top 3 losers
            {"type": "most_traded", "data": data["most_actively_traded"][:3]}  # Top 3 traded
        ]
        market_insights = market_analyst.execute_task(
            task_description="Analyze the following market data.",
            additional_data=market_data
        )
        st.write("Market Analyst Insights:", market_insights)
        return market_insights
    else:
        st.error("Invalid data format received from API.")
        return None

### Step 3: Combine and Send to Risk Analyst ###

def send_to_risk_analyst(news_insights, market_insights):
    """Send insights from Researcher and Market Analyst to Risk Analyst."""
    if news_insights and market_insights:
        combined_insights = [
            {"role": "Researcher", "content": news_insights},
            {"role": "Market Analyst", "content": market_insights}
        ]
        risk_insights = risk_analyst.execute_task(
            task_description="Evaluate risks based on the following insights.",
            additional_data=combined_insights
        )
        st.write("Risk Analyst Insights:", risk_insights)
        return risk_insights
    else:
        st.error("Missing insights from Researcher or Market Analyst.")
        return None

### Step 4: Generate Newsletter with Writer ###

def generate_sequential_newsletter(news_insights, market_insights, risk_insights):
    """Generate a newsletter using outputs from all three agents."""
    if news_insights and market_insights and risk_insights:
        combined_data = [
            {"role": "Researcher", "content": news_insights},
            {"role": "Market Analyst", "content": market_insights},
            {"role": "Risk Analyst", "content": risk_insights}
        ]
        writer_task_description = "Write a cohesive newsletter based on insights from news, market trends, and risk analysis."
        newsletter = writer.execute_task(writer_task_description, additional_data=combined_data)

        # Display the newsletter
        if "Error" in newsletter:
            st.error("Failed to generate the newsletter.")
        else:
            st.subheader("Generated Newsletter")
            st.markdown(f"## Writer's Newsletter\n{newsletter}")
    else:
        st.error("Missing insights for generating the newsletter.")
        
def factcheck_with_bespoke(claim, context):
    """Perform fact-checking using Bespoke Labs."""
    try:
        response = bl.minicheck.factcheck.create(claim=claim, context=context)
        return {
            "support_prob": getattr(response, "support_prob", "N/A"),
            "details": str(response)
        }
    except Exception as e:
        st.error(f"Error with Bespoke Labs Fact-Check: {e}")
        return None

def retrieve_from_multiple_rags(query, collections, top_k=3):
    """Search multiple collections for relevant RAG data."""
    results = []
    for collection_name in collections:
        collection_results = retrieve_from_chromadb(collection_name, query, top_k)
        results.extend([doc for doc in collection_results if doc])
    return results

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
        st.write("News API Response:", data)
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
        st.write("Ticker Trends API Response:", data)
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
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        return content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI GPT-4: {e}")
        return "I'm sorry, I couldn't process your request at this time."

def fetch_ticker_price(ticker):
    """Fetch the latest price for the given ticker."""
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={alpha_vantage_key}"
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
            return {"error": "Invalid ticker or no data available."}
    except Exception as e:
        return {"error": f"Error fetching ticker price: {e}"}

### Visualization ###
def plot_top_gainers(gainers):
    """Visualize top gainers using a bar chart."""
    tickers = [g['symbol'] for g in gainers]
    gains = [float(g['percent_change']) for g in gainers]

    plt.bar(tickers, gains)
    plt.title("Top Gainers")
    plt.xlabel("Stocks")
    plt.ylabel("Percentage Gain")
    st.pyplot(plt)

class RAGAgent:
    def __init__(self, role, goal):
        self.role = role
        self.goal = goal

    def execute_task(self, task_description, additional_data=None):
        if "news" in self.goal.lower():
            retrieved_data = retrieve_from_chromadb("news_sentiment_data", task_description)
        elif "trends" in self.goal.lower():
            retrieved_data = retrieve_from_chromadb("ticker_trends_data", task_description)
        else:
            retrieved_data = []

        combined_data = retrieved_data
        if additional_data:
            combined_data.extend(additional_data)

        augmented_prompt = (
            f"Role: {self.role}\nGoal: {self.goal}\n"
            f"Task: {task_description}\nRelevant Data:\n{json.dumps(combined_data)}\n"
            "Please provide a detailed response of up to 1,800 tokens. Focus on key metrics, trends, and actionable insights."
        )
        result = call_openai_gpt4(augmented_prompt)
        return result

### Agents and Tasks ###
researcher = RAGAgent(role="Researcher", goal="Process news data")
market_analyst = RAGAgent(role="Market Analyst", goal="Analyze trends")
risk_analyst = RAGAgent(role="Risk Analyst", goal="Identify risks")
writer = RAGAgent(role="Writer", goal="Generate newsletter")

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
            "using appropriate financial jargon and terminologies. "
            "Focus on key metrics, trends, risks, and actionable insights, ensuring the output is concise, "
            "informative, and approximately 2,000 tokens in length."
        )
        newsletter = writer.execute_task(writer_task_description, additional_data=[combined_context])

        if "Error" in newsletter:
            st.error("Failed to generate the newsletter.")
        else:
            st.subheader("Generated Newsletter")
            st.markdown(f"## Writer's Newsletter\n{newsletter}")
    else:
        st.error("Missing insights for generating the newsletter.")

### Main Buttons ###
if st.button("Fetch and Store News Data"):
    fetch_and_update_news_data()

if st.button("Fetch and Store Trends Data"):
    fetch_and_update_ticker_trends_data()

if st.button("Generate Newsletter"):
    # Step 1: Fetch and process news data
    st.write("Executing: Extract insights from news data (Researcher)")
    news_insights = researcher.execute_task("Extract insights from news data")

    # Step 2: Fetch and process market data
    st.write("Executing: Analyze market trends (Market Analyst)")
    market_insights = market_analyst.execute_task("Analyze market trends")

    # Step 3: Combine and send to Risk Analyst
    st.write("Executing: Analyze risk data (Risk Analyst)")
    risk_insights = risk_analyst.execute_task("Analyze risk data", additional_data=[news_insights, market_insights])

    # Step 4: Generate newsletter using all insights
    generate_sequential_newsletter(news_insights, market_insights, risk_insights)

### Chatbot ###
st.subheader("Chatbot")

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

user_input = st.text_input("Ask me something:")

if st.button("Send"):
    if not user_input.strip():
        st.write("Please enter a query.")
    else:
        if user_input.isalpha() and len(user_input) <= 5:  # Check for ticker symbol
            ticker_data = fetch_ticker_price(user_input.upper())
            if "error" in ticker_data:
                response = f"Error: {ticker_data['error']}"
            else:
                response = (
                    f"**Ticker:** {ticker_data['ticker']}\n"
                    f"**Date:** {ticker_data['date']}\n"
                    f"**Open:** {ticker_data['open']}\n"
                    f"**High:** {ticker_data['high']}\n"
                    f"**Low:** {ticker_data['low']}\n"
                    f"**Close:** {ticker_data['close']}\n"
                    f"**Volume:** {ticker_data['volume']}\n"
                )
            st.write(response)
        else:
            rag_results = retrieve_from_multiple_rags(user_input, ["news_sentiment_data", "ticker_trends_data"])
            if rag_results:
                # Ensure each item is a string before joining
                context = "\n".join([str(doc) for doc in rag_results])
            else:
                context = "No relevant data found."
            
            prompt = f"User: {user_input}\nContext: {context}"
            response = call_openai_gpt4(prompt)
            st.write(response)

        st.session_state["conversation_history"].append({"user": user_input, "bot": response})

