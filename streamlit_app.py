import streamlit as st
import requests
import json
import openai
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb


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

def retrieve_from_multiple_rags(query, collections, top_k=5):
    """Search multiple collections for relevant RAG data."""
    results = []
    for collection_name in collections:
        collection_results = retrieve_from_chromadb(collection_name, query, top_k)
        results.extend([doc for doc in collection_results if doc])  # Filter empty results
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

def retrieve_from_chromadb(collection_name, query, top_k=5):
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
        # Correctly access the content using dot notation
        content = response.choices[0].message["content"]
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

### RAG-Agent Definition ###

class RAGAgent:
    def __init__(self, role, goal):
        self.role = role
        self.goal = goal

    def execute_task(self, task_description, additional_data=None):
        """Execute the task using RAG and GPT-4 summarization."""
        # Retrieve relevant data from ChromaDB
        if "news" in self.goal.lower():
            retrieved_data = retrieve_from_chromadb("news_sentiment_data", task_description)
        elif "trends" in self.goal.lower():
            retrieved_data = retrieve_from_chromadb("ticker_trends_data", task_description)
        else:
            retrieved_data = []

        # Combine additional data if provided
        combined_data = retrieved_data
        if additional_data:
            combined_data.extend(additional_data)

        # Combine retrieved data with task description
        augmented_prompt = f"Role: {self.role}\nGoal: {self.goal}\nTask: {task_description}\nRelevant Data:\n{json.dumps(combined_data)}"

        # Call GPT-4 for summarization
        summary = call_openai_gpt4(augmented_prompt)
        return summary

### Agents and Tasks ###

researcher = RAGAgent(role="Researcher", goal="Process news data")
market_analyst = RAGAgent(role="Market Analyst", goal="Analyze trends")
risk_analyst = RAGAgent(role="Risk Analyst", goal="Identify risks")
writer = RAGAgent(role="Writer", goal="Generate newsletter")

tasks = [
    {"description": "Extract insights from news data", "agent": researcher},
    {"description": "Analyze market trends", "agent": market_analyst},
    {"description": "Analyze risk data", "agent": risk_analyst},
    {"description": "Write the newsletter", "agent": writer},
]

### Newsletter Generation ###

def generate_newsletter_with_rag():
    """Generate the newsletter using RAG and agents."""
    newsletter_content = []

    # Step 1: Execute tasks for Risk Analyst, Market Analyst, and Researcher
    st.write("Executing: Extract insights from news data (Researcher)")
    news_results = researcher.execute_task("Extract insights from news data")

    st.write("Executing: Analyze market trends (Market Analyst)")
    trends_results = market_analyst.execute_task("Analyze market trends")

    st.write("Executing: Analyze risk data (Risk Analyst)")
    risk_results = risk_analyst.execute_task("Analyze risk data")

    # Step 2: Combine insights for Writer Agent
    combined_data = [
        {"role": "Researcher", "content": news_results},
        {"role": "Market Analyst", "content": trends_results},
        {"role": "Risk Analyst", "content": risk_results}
    ]

    # Step 3: Generate the newsletter with Writer Agent
    st.write("Executing: Write the newsletter (Writer)")
    writer_task_description = "Write a cohesive newsletter based on insights from news, market trends, and risk analysis."
    newsletter = writer.execute_task(writer_task_description, additional_data=combined_data)

    # Step 4: Display the generated newsletter
    if "Error" in newsletter:
        st.error("Failed to generate the newsletter.")
    else:
        newsletter_content.append(f"## Writer's Newsletter\n{newsletter}\n")

    st.subheader("Generated Newsletter")
    st.markdown("\n".join(newsletter_content))

### Main Page Buttons ###
if st.button("Fetch and Store News Data"):
    fetch_and_update_news_data()

if st.button("Fetch and Store Trends Data"):
    fetch_and_update_ticker_trends_data()

if st.button("Generate Newsletter"):
    generate_newsletter_with_rag()

### Chatbot UI ###
st.subheader("Chatbot")

user_input = st.text_input("Ask me something:")

if st.button("Send"):
    if len(user_input.strip()) == 0:
        st.write("Please enter a query.")
    else:
        user_input = user_input.strip()
        
        # Step 1: Check if it's a ticker query
        if user_input.isalpha() and len(user_input) <= 5:  # Assuming stock tickers are alphabetic and <= 5 characters
            st.write(f"Fetching daily information for ticker: {user_input.upper()}...")
            ticker_result = fetch_ticker_price(user_input.upper())
            if "error" in ticker_result:
                st.error(ticker_result["error"])
            else:
                st.write(
                    f"**Ticker:** {ticker_result['ticker']}\n"
                    f"**Date:** {ticker_result['date']}\n"
                    f"**Open:** {ticker_result['open']}\n"
                    f"**High:** {ticker_result['high']}\n"
                    f"**Low:** {ticker_result['low']}\n"
                    f"**Close:** {ticker_result['close']}\n"
                    f"**Volume:** {ticker_result['volume']}\n"
                )
        else:
            # Step 2: Query RAG and Use OpenAI GPT-4 for Contextual Understanding
            st.write("Searching in stored RAG data...")
            rag_collections = ["news_sentiment_data", "ticker_trends_data"]
            rag_results = retrieve_from_multiple_rags(user_input, rag_collections)

            if rag_results:
                # Combine RAG results into a context for GPT-4
                st.write("Found relevant data in stored RAG. Passing to GPT-4 for contextual understanding...")
                # Safely process rag_results into a string context
                context = "\n".join(
                    [json.dumps(result, indent=2) if isinstance(result, dict) else str(result) for result in rag_results]
                )
                prompt = (
                    f"You are a helpful assistant. Based on the user's query below, and the context from stored data, "
                    f"provide a well-framed and relevant response:\n\n"
                    f"Query: {user_input}\n\n"
                    f"Context from RAG:\n{context}"
                )
                response = call_openai_gpt4(prompt)
                st.write(response)
            else:
                # Fallback to OpenAI GPT-4
                st.write("Not found in today's data. Searching online...")
                prompt = (
                    f"You are a helpful assistant. Based on the user's query below, provide a well-framed and "
                    f"relevant response:\n\nQuery: {user_input}"
                )
                response = call_openai_gpt4(prompt)
                st.write(response)
