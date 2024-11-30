import streamlit as st
import requests
import json
import openai
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from bespokelabs import BespokeLabs
import bespokelabs

# Initialize Bespoke Labs API
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
st.title("Alpha Vantage Multi-Agent System with RAG and OpenAI GPT-4")

### Helper Functions ###
def validate_with_bespoke(newsletter, context):
    """
    Validate the newsletter using Bespoke Labs with enhanced error handling.
    :param newsletter: The generated newsletter content (claim).
    :param context: The context from RAG data (serialized JSON).
    :return: Support probability and status message.
    """
    try:
        st.write("Validating the newsletter with Bespoke Labs...")
        factcheck_response = bl.minicheck.factcheck.create(
            claim=newsletter,
            context=json.dumps(context)
        )
        support_prob = factcheck_response.get("support_prob", "N/A")
        if support_prob == "N/A":
            return None, "Validation returned no support probability."
        elif support_prob >= 0.8:
            return support_prob, "The newsletter is highly supported by the context."
        elif support_prob >= 0.5:
            return support_prob, "The newsletter has partial support from the context."
        else:
            return support_prob, "The newsletter lacks sufficient support from the context."
    except bespokelabs.APIConnectionError as e:
        st.error("The server could not be reached. Please check your network connection.")
        return None, f"Connection error: {e.__cause__}"
    except bespokelabs.RateLimitError as e:
        st.error("Rate limit exceeded. Please try again later.")
        return None, "Rate limit exceeded."
    except bespokelabs.APIStatusError as e:
        st.error(f"API returned a non-success status code: {e.status_code}")
        st.error(f"Response details: {e.response}")
        return None, f"API error with status code {e.status_code}: {e.response}"
    except bespokelabs.APIError as e:
        st.error("An unexpected API error occurred.")
        return None, f"Unexpected API error: {e}"
    except Exception as e:
        st.error("An unknown error occurred.")
        return None, f"Unknown error: {e}"


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
        # Correctly access the content attribute using dot notation
        content = response.choices[0].message.content
        return content.strip()
    except Exception as e:
        st.error(f"Error calling OpenAI GPT-4: {e}")
        return "I'm sorry, I couldn't process your request at this time."


### Newsletter Generation ###
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


researcher = RAGAgent(role="Researcher", goal="Process news data")
market_analyst = RAGAgent(role="Market Analyst", goal="Analyze trends")
risk_analyst = RAGAgent(role="Risk Analyst", goal="Identify risks")
writer = RAGAgent(role="Writer", goal="Generate newsletter")


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
        return None, combined_data
    else:
        newsletter_content.append(f"## Writer's Newsletter\n{newsletter}\n")
        st.subheader("Generated Newsletter")
        st.markdown("\n".join(newsletter_content))
        return newsletter, combined_data


### Main Page Buttons ###
if st.button("Fetch and Store News Data"):
    fetch_and_update_news_data()

if st.button("Fetch and Store Trends Data"):
    fetch_and_update_ticker_trends_data()

if st.button("Generate Newsletter"):
    newsletter, rag_context = generate_newsletter_with_rag()

if st.button("Validate Newsletter with Bespoke Labs"):
    if 'newsletter' not in locals() or newsletter is None:
        st.error("Please generate the newsletter first.")
    else:
        support_prob, message = validate_with_bespoke(newsletter, rag_context)
        if support_prob is not None:
            st.write(f"Newsletter Fact-Check Support Probability: {support_prob}")
            if support_prob >= 0.8:
                st.success(message)
            elif support_prob >= 0.5:
                st.warning(message)
            else:
                st.error(message)
        else:
            st.error(message)
