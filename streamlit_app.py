import streamlit as st
import requests
import json
import openai
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from bespokelabs import BespokeLabs
import pandas as pd

# Initialize Bespoke Labs with the API key
bl = BespokeLabs(auth_token=st.secrets["bespoke"]["api_key"])

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# Access keys from secrets.toml
alpha_vantage_key = st.secrets["alpha_vantage"]["api_key"]
openai.api_key = st.secrets["openai"]["api_key"]

# Streamlit App Title
st.title("Financial Newsletter generation using Multi-Agent System")

### Helper Functions ###
### Helper Functions ###
### Helper Functions ###
def is_ticker_query(user_input):
    """Determine if the user query is a ticker symbol."""
    return user_input.isalpha() and len(user_input) <= 5  

def prettify_openai_response(prompt, retrieved_data):
    """Use OpenAI to prettify RAG-based answers."""
    try:
        # Combine context and user query for OpenAI processing
        context = "\n".join([f"- {doc}" for doc in retrieved_data])
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial assistant. Use the context to answer the query."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error prettifying response: {e}"

def retrieve_from_rag(query, collections=["news_sentiment_data", "ticker_trends_data"]):
    """Retrieve relevant information from RAG."""
    results = []
    for collection_name in collections:
        try:
            collection = client.get_or_create_collection(collection_name)
            response = collection.query(query_texts=[query], n_results=5)
            results.extend(response.get("documents", []))
        except Exception as e:
            st.error(f"Error retrieving data from {collection_name}: {e}")
    return results

def handle_fallback_with_openai(prompt):
    """Fallback to OpenAI if no specific handler is available."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error with OpenAI fallback: {e}"

def retrieve_from_multiple_rags(query, collections, top_k=3):
    """Search multiple collections for relevant RAG data."""
    results = []
    for collection_name in collections:
        try:
            collection_results = retrieve_from_chromadb(collection_name, query, top_k)
            results.extend(collection_results)  # Combine results from all collections
        except Exception as e:
            st.error(f"Error retrieving data from collection {collection_name}: {e}")
    return results

tools = [
    {
        "name": "fetch_ticker_price",  # The required 'name' parameter
        "description": "Fetch the latest price for a given stock ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol (e.g., AAPL, MSFT)."
                }
            },
            "required": ["ticker"]
        }
    }
]

def format_ticker_response(ticker_data):
    """Format the ticker data for clean and readable output."""
    if "error" in ticker_data:
        return f"Error: {ticker_data['error']}"

    # Properly formatted plain text response
    return (
        f"Ticker: {ticker_data['ticker']}\n"
        f"Date: {ticker_data['date']}\n"
        f"Open Price: ${float(ticker_data['open']):,.2f}\n"
        f"High Price: ${float(ticker_data['high']):,.2f}\n"
        f"Low Price: ${float(ticker_data['low']):,.2f}\n"
        f"Close Price: ${float(ticker_data['close']):,.2f}\n"
        f"Volume: {int(ticker_data['volume']):,} shares"
    )

### Define Function ###
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

### Tool Handler ###
def handle_tool_call(tool_name, parameters):
    """Handle tool calls dynamically based on the tool schema."""
    if tool_name == "fetch_ticker_price":
        ticker = parameters.get("ticker")
        if not ticker:
            return {"error": "Ticker symbol is required for this function."}
        return fetch_ticker_price(ticker)
    else:
        return {"error": f"Tool '{tool_name}' is not implemented."}

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

def update_chromadb(collection_name, data):
    """Update ChromaDB with new data."""
    collection = client.get_or_create_collection(collection_name)
    try:
        for i, item in enumerate(data, start=1):
            collection.add(
                ids=[f"{collection_name}_{i}"],
                metadatas=[{"source": item.get("source", "N/A"), "time_published": item.get("time_published", "N/A")}],
                documents=[json.dumps(item)]
            )
        st.success(f"Updated {collection_name} with new data.")
    except Exception as e:
        st.error(f"Error updating {collection_name}: {e}")

def fetch_and_update_news_data():
    """Fetch news data from Alpha Vantage and update ChromaDB."""
    news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={alpha_vantage_key}&limit=50"
    try:
        response = requests.get(news_url)
        response.raise_for_status()
        data = response.json()
        if "feed" in data:
            update_chromadb("news_sentiment_data", data["feed"])
        else:
            st.error("Invalid news data format.")
    except Exception as e:
        st.error(f"Error fetching news data: {e}")

def fetch_and_update_market_data():
    """Fetch market data from Alpha Vantage and update ChromaDB."""
    market_url = f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={alpha_vantage_key}"
    try:
        response = requests.get(market_url)
        response.raise_for_status()
        data = response.json()
        if "top_gainers" in data:
            combined_data = [
                {"type": "top_gainers", "data": data["top_gainers"]},
                {"type": "top_losers", "data": data["top_losers"]},
                {"type": "most_actively_traded", "data": data["most_actively_traded"]}
            ]
            update_chromadb("market_data", combined_data)
        else:
            st.error("Invalid market data format.")
    except Exception as e:
        st.error(f"Error fetching market data: {e}")

### RAGAgent ###
class RAGAgent:
    def __init__(self, role, goal):
        self.role = role
        self.goal = goal

    def execute_task(self, task_description, additional_data=None):
        # Fetch relevant data from ChromaDB collections based on the role
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

### Agents ###
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
            "Role: Writer\n"
            "Goal: Generate a professional financial newsletter.\n"
            "Task: Write a comprehensive newsletter based on the provided insights. Ensure that:\n\n"
            "1. All statements are explicitly supported by the provided context.\n"
            "2. No additional assumptions, extrapolations, or speculative language are included.\n"
            "3. The newsletter uses financial jargon and concise, fact-based reporting.\n"
            "4. Avoid editorializing or introducing opinions.\n"
            "5. Each claim in the newsletter should have a direct connection to the data provided.\n\n"
            "Use the following structure for clarity:\n"
            "- **Introduction**: Briefly summarize the key themes (e.g., market trends, risks, technological developments).\n"
            "- **Section 1**: Detailed analysis of news-related insights (from the Researcher).\n"
            "- **Section 2**: Market trends and key takeaways (from the Market Analyst).\n"
            "- **Section 3**: Risk assessments and implications (from the Risk Analyst).\n"
            "- **Conclusion**: Highlight actionable insights and their potential impact on financial strategies.\n\n"
            "Ensure that the newsletter is concise, professional, and adheres to a maximum of 2,000 tokens.\n"
            f"Relevant Data:\n{combined_context}"
        )
        newsletter = writer.execute_task(writer_task_description, additional_data=[combined_context])

        if "Error" in newsletter:
            st.error("Failed to generate the newsletter.")
        else:
            st.session_state["newsletter_content"] = newsletter  # Store newsletter for fact-checking
            st.session_state["newsletter_context"] = combined_context  # Store input context
            st.subheader("Generated Newsletter")
            st.markdown(f"## Writer's Newsletter\n{newsletter}")
    else:
        st.error("Missing insights for generating the newsletter.")

### Fact-Check with Bespoke Labs ###
def factcheck_with_bespoke_from_newsletter():
    """Fact-check using the newsletter content as claim and its input as context."""
    newsletter_content = st.session_state.get("newsletter_content", None)
    newsletter_context = st.session_state.get("newsletter_context", None)

    if not newsletter_content or not newsletter_context:
        st.error("No newsletter content or context available for fact-checking. Generate a newsletter first.")
        return None

    try:
        response = bl.minicheck.factcheck.create(claim=newsletter_content, context=newsletter_context)
        return {
            "support_prob": getattr(response, "support_prob", "N/A"),
            "details": str(response)
        }
    except Exception as e:
        st.error(f"Error with Bespoke Labs Fact-Check: {e}")
        return None

### Buttons for Updating Data ###
if st.button("Update News Data"):
    fetch_and_update_news_data()

if st.button("Update Market Data"):
    fetch_and_update_market_data()

### Generate Newsletter Button ###
if st.button("Generate Newsletter"):
    st.write("Generating newsletter...")
    news_insights = researcher.execute_task("Extract insights from news data")
    market_insights = market_analyst.execute_task("Analyze market trends")
    risk_insights = risk_analyst.execute_task("Identify risks", additional_data=[news_insights, market_insights])
    generate_sequential_newsletter(news_insights, market_insights, risk_insights)

### Fact-Check Newsletter Button ###
if st.button("Fact-Check Newsletter"):
    factcheck_result = factcheck_with_bespoke_from_newsletter()
    if factcheck_result:
        st.write("**Fact-Check Result**")
        st.write(f"Support Probability: {factcheck_result['support_prob']}")
        st.write(f"Details: {factcheck_result['details']}")

### CSV Upload ###
st.subheader("Upload Investor Portfolio")

uploaded_file = st.file_uploader("Upload a CSV file with the following columns: Investor Name, Company, Invested Amount, Returns, Returns (%), Ticker", type=["csv"])

if uploaded_file:
    # Read the CSV file
    portfolio_df = pd.read_csv(uploaded_file)

    # Display the uploaded portfolio
    st.write("Uploaded Portfolio:")
    st.dataframe(portfolio_df)

    # Fetch stock data for each ticker
    st.write("Fetching stock data...")
    stock_data = []
    for ticker in portfolio_df["Ticker"]:
        ticker_data = fetch_ticker_price(ticker)
        stock_data.append(ticker_data)

    # Process the data for recommendations
    valid_stocks = [data for data in stock_data if "error" not in data]
    if valid_stocks:
        st.json(valid_stocks)

        # Use OpenAI to analyze and recommend actions
        analysis_prompt = (
            f"Analyze the following stock portfolio data and provide investment recommendations in 250 words. "
            f"Suggest which stocks to hold, buy more of, or sell. Use financial insights keep this within 100 words an precise.\n\n"
            f"Stock Data: {json.dumps(valid_stocks, indent=2)}"
        )
        recommendations = call_openai_gpt4(analysis_prompt)
        st.subheader("Investment Recommendations")
        st.markdown(recommendations)
    else:
        st.error("No valid stock data retrieved. Please check the tickers in your file.")

### Chatbot ###
st.subheader("Chatbot")

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

user_input = st.text_input("Ask me something:")

if st.button("Send"):
    if len(user_input.strip()) == 0:
        st.write("Please enter a query.")
    else:
        user_input = user_input.strip()

        # Case 1: Match with tool
        for tool in tools:
            if tool["name"] == "fetch_ticker_price" and is_ticker_query(user_input):
                parameters = {"ticker": user_input.upper()}
                tool_result = handle_tool_call(tool["name"], parameters)
                bot_response = format_ticker_response(tool_result)
                st.text(bot_response)  # Use plain text for ticker responses
                break
        else:
            # Case 2: Default to RAG or OpenAI
            rag_collections = ["news_sentiment_data", "ticker_trends_data"]
            rag_results = retrieve_from_multiple_rags(user_input, rag_collections)

            if rag_results:
                # Combine RAG results into a context for GPT-4
                context = "\n".join(
                    [json.dumps(result, indent=2) if isinstance(result, dict) else str(result) for result in rag_results]
                )
                prompt = (
                    f"You are a helpful assistant. Below is the user's query and context. "
                    f"Provide a relevant and well-framed response.\n\n"
                    f"Query: {user_input}\n\n"
                    f"Context:\n{context}"
                )
                bot_response = call_openai_gpt4(prompt)
                st.markdown(bot_response)  # Use Markdown for GPT-4-prettified RAG responses
            else:
                # Fallback to OpenAI GPT-4
                st.write("No relevant data found in RAG. Falling back to OpenAI...")
                bot_response = call_openai_gpt4(user_input)
                st.markdown(bot_response)

        # Update Conversation History
        st.session_state["conversation_history"].append({"user": user_input, "bot": bot_response})
