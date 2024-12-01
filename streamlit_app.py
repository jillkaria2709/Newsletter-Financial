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

def retrieve_from_multiple_rags(query, collections, top_k=3):
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
            model="gpt-4o-mini",  # Ensure this is the correct model name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        # Access the content of the first choice correctly
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

class RAGAgent:
    def __init__(self, role, goal):
        self.role = role
        self.goal = goal

    def execute_task(self, task_description, additional_data=None):
        """Execute the task using RAG data and OpenAI GPT-4 for formatting."""
        # Retrieve relevant data from ChromaDB
        if "news" in self.goal.lower():
            retrieved_data = retrieve_from_chromadb("news_sentiment_data", task_description)
        elif "trends" in self.goal.lower():
            retrieved_data = retrieve_from_chromadb("ticker_trends_data", task_description)
        else:
            retrieved_data = []

        # Combine retrieved data with additional context if provided
        combined_data = retrieved_data
        if additional_data:
            combined_data.extend(additional_data)

        # Ensure only RAG data is passed to GPT-4
        formatted_data = json.dumps(combined_data, indent=2)

        # Construct the prompt for GPT-4
        prompt = (
            f"Role: {self.role}\n"
            f"Goal: {self.goal}\n"
            f"Task: {task_description}\n\n"
            f"Use ONLY the following RAG data to frame your response:\n"
            f"{formatted_data}\n\n"
            f"Do not add any new information not present in the RAG data."
        )

        # Use OpenAI GPT-4 to frame the response
        response = call_openai_gpt4(prompt)
        return response

### Newsletter Generation ###

def generate_newsletter_with_rag():
    """Generate the newsletter using RAG and agents and validate it with Bespoke Labs."""
    newsletter_content = []

    # Step 1: Execute Tasks for Each Agent
    st.write("Executing: Extract insights from news data (Researcher)")
    news_results = researcher.execute_task("Extract insights from news data")

    st.write("Executing: Analyze market trends (Market Analyst)")
    trends_results = market_analyst.execute_task("Analyze market trends")

    st.write("Executing: Analyze risk data (Risk Analyst)")
    risk_results = risk_analyst.execute_task("Analyze risk data")

    # Step 2: Combine Agent Outputs as Context
    combined_data = [
        {"role": "Researcher", "content": news_results},
        {"role": "Market Analyst", "content": trends_results},
        {"role": "Risk Analyst", "content": risk_results}
    ]

    # Step 3: Write the Newsletter (Writer Agent)
    st.write("Executing: Write the newsletter (Writer)")
    writer_task_description = (
        "Write a cohesive newsletter using ONLY the provided insights from RAG data. "
        "Do not include any new or external information."
    )
    newsletter = writer.execute_task(writer_task_description, additional_data=combined_data)

    # Step 4: Display the Generated Newsletter
    if "Error" in newsletter:
        st.error("Failed to generate the newsletter.")
    else:
        newsletter_content.append(f"## Writer's Newsletter\n{newsletter}\n")
        st.subheader("Generated Newsletter")
        st.markdown("\n".join(newsletter_content))

    # Step 5: Validate the Newsletter with Bespoke Labs
    try:
        st.write("Validating the newsletter with Bespoke Labs...")
        
        # Combine all agent outputs for the context
        context = {agent["role"]: agent["content"] for agent in combined_data}
        
        # Debugging: Display the context passed to Bespoke Labs
        st.write("Validation Context:", context)

        factcheck_response = bl.minicheck.factcheck.create(
            claim=newsletter,  # Use the generated newsletter as the claim
            context=json.dumps(context)  # Use combined agent outputs as the context
        )

        # Step 6: Display the Factcheck Response
        st.write("Factcheck Response (Raw):", factcheck_response)
        support_prob = getattr(factcheck_response, "support_prob", None)

        if support_prob is None:
            st.error("Validation returned no support probability.")
        else:
            st.write(f"Newsletter Fact-Check Support Probability: {support_prob}")
            if support_prob >= 0.7:
                st.success("The newsletter is well-supported by the context.")
            elif support_prob >= 0.4:
                st.warning("The newsletter has partial support from the context.")
            else:
                st.error("The newsletter lacks sufficient support from the context.")
                st.warning("Consider refining the newsletter with more context-relevant data. Revisit the extracted insights for better alignment.")
    except AttributeError:
        st.error("The 'support_prob' attribute is missing in the response.")
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
