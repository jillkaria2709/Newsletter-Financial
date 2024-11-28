import streamlit as st
import requests
import json
__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from openai import OpenAI

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient()

# API key and URLs for Alpha Vantage
api_key = st.secrets["api_keys"]["alpha_vantage"]
news_url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={api_key}&limit=50'
tickers_url = f'https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS&apikey={api_key}'

# Initialize OpenAI Client
openai_client = OpenAI(api_key=st.secrets["api_keys"]["openai"])

# Streamlit App Title
st.title("Alpha Vantage RAG System")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.radio(
    "Choose an action:",
    ["Load News Data", "Retrieve News Data", "Load Ticker Trends Data", "Retrieve Ticker Trends Data", "Generate Newsletter"]
)

# Initialize ChromaDB Collections
news_collection = client.get_or_create_collection("news_sentiment_data")
ticker_collection = client.get_or_create_collection("ticker_trends_data")

### Function to Load News Data into ChromaDB ###
def load_news_data():
    try:
        response = requests.get(news_url)
        response.raise_for_status()
        data = response.json()

        if 'feed' in data:
            for i, item in enumerate(data['feed'], start=1):
                document = {
                    "id": str(i),
                    "title": item["title"],
                    "url": item["url"],
                    "time_published": item["time_published"],
                    "source": item["source"],
                    "summary": item.get("summary", "N/A"),
                    "topics": [topic["topic"] for topic in item.get("topics", [])],
                    "overall_sentiment_label": item.get("overall_sentiment_label", "N/A"),
                    "overall_sentiment_score": item.get("overall_sentiment_score", "N/A")
                }

                # Add to ChromaDB
                news_collection.add(
                    ids=[document["id"]],
                    metadatas=[{
                        "title": document["title"],
                        "source": document["source"],
                        "time_published": document["time_published"],
                        "overall_sentiment": document["overall_sentiment_label"]
                    }],
                    documents=[json.dumps(document)]
                )

            st.success(f"Loaded {len(data['feed'])} news items into ChromaDB.")
        else:
            st.error("No news data found.")
    except Exception as e:
        st.error(f"Error loading news data: {e}")

### Function to Retrieve News Data from ChromaDB ###
def retrieve_news_data():
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

### Function to Load Ticker Trends Data into ChromaDB ###
def load_ticker_trends_data():
    try:
        response = requests.get(tickers_url)
        response.raise_for_status()
        data = response.json()

        if "top_gainers" in data and "top_losers" in data:
            ticker_collection.add(
                ids=["top_gainers"],
                documents=[json.dumps(data["top_gainers"])],
                metadatas=[{"type": "Top Gainers"}]
            )
            ticker_collection.add(
                ids=["top_losers"],
                documents=[json.dumps(data["top_losers"])],
                metadatas=[{"type": "Top Losers"}]
            )
            st.success("Ticker trends data loaded into ChromaDB.")
        else:
            st.error("Invalid data format from API.")
    except Exception as e:
        st.error(f"Error loading ticker trends data: {e}")

### Function to Retrieve Ticker Trends Data ###
def retrieve_ticker_trends_data():
    data_type = st.radio(
        "Data Type", ["Top Gainers", "Top Losers"]
    )

    data_id_mapping = {
        "Top Gainers": "top_gainers",
        "Top Losers": "top_losers"
    }

    if st.button("Retrieve Ticker Trends Data"):
        try:
            results = ticker_collection.get(ids=[data_id_mapping[data_type]])
            if results["documents"]:
                for document, metadata in zip(results["documents"], results["metadatas"]):
                    st.write(f"### {metadata['type']}")
                    st.json(json.loads(document))
            else:
                st.warning("No data found.")
        except Exception as e:
            st.error(f"Error retrieving ticker trends data: {e}")

def truncate_content(content, max_chars=4000):
    """
    Truncate content to a maximum number of characters.
    """
    if len(content) > max_chars:
        return content[:max_chars] + "\n\n[Content truncated...]"
    return content

def summarize_content(content, role_description="Summarize the following content:"):
    """
    Dynamically summarize content to fit within token limits.
    """
    try:
        # Check if content exceeds the token limit for the OpenAI model
        max_tokens_per_request = 8192  # Token limit for GPT-4
        chunk_size = 2000  # Arbitrary chunk size for splitting
        content_tokens = len(content.split())  # Approximation of token count

        if content_tokens > max_tokens_per_request - 2000:
            # Dynamically split content into smaller chunks
            chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
            chunk_summaries = []

            for chunk in chunks:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful summarizer."},
                        {"role": "user", "content": f"{role_description}\n{chunk}"}
                    ]
                )
                chunk_summaries.append(response.choices[0].message.content)

            # Summarize the combined chunk summaries
            combined_summary = "\n".join(chunk_summaries)
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful summarizer."},
                    {"role": "user", "content": f"Summarize the following combined content:\n{combined_summary}"}
                ]
            )
            return response.choices[0].message.content
        else:
            # Summarize directly if content is within token limits
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful summarizer."},
                    {"role": "user", "content": f"{role_description}\n{content}"}
                ]
            )
            return response.choices[0].message.content

    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return "Error during summarization."

def generate_newsletter():
    try:
        # Query and process news collection - Limit to Top 5
        news_results = news_collection.query(query_texts=[""], n_results=5)
        raw_news_content = "\n".join(
            [
                f"{doc.get('title', 'No Title')}: {doc.get('summary', 'No Summary')}\n"
                f"Source: {doc.get('source', 'Unknown')}\n"
                f"Published: {doc.get('time_published', 'Unknown')}\n"
                f"Sentiment: {doc.get('overall_sentiment_label', 'Unknown')} "
                f"(Score: {doc.get('overall_sentiment_score', 'N/A')})\n"
                f"Topics: {', '.join(doc.get('topics', []))}"
                if isinstance(doc, dict) else
                f"Unstructured Document: {doc}"
                for doc in news_results["documents"]
            ]
        )
        summarized_news = summarize_content(
            raw_news_content, 
            "Summarize the following top 5 news items for a financial newsletter:"
        )

        # Query and process ticker trends collection
        ticker_results = ticker_collection.query(query_texts=[""], n_results=1000)

        # Extract top 5 gainers and losers
        gainers = [
            doc for doc, meta in zip(ticker_results["documents"], ticker_results["metadatas"])
            if meta.get("type") == "top_gainers"
        ]
        losers = [
            doc for doc, meta in zip(ticker_results["documents"], ticker_results["metadatas"])
            if meta.get("type") == "top_losers"
        ]

        # Sort and take top 5 by change_percentage
        top_5_gainers = sorted(
            gainers,
            key=lambda x: float(x.get("change_percentage", 0).replace("%", "")),
            reverse=True
        )[:5]
        top_5_losers = sorted(
            losers,
            key=lambda x: float(x.get("change_percentage", 0).replace("%", "")),
            reverse=True
        )[:5]

        # Prepare raw ticker content
        raw_ticker_content = "\n".join(
            [
                f"Gainer: {gainer.get('ticker', 'Unknown')} - Change: {gainer.get('change_percentage', 'N/A')} "
                f"(Price: {gainer.get('price', 'N/A')}, Volume: {gainer.get('volume', 'N/A')})"
                for gainer in top_5_gainers
            ] + [
                f"Loser: {loser.get('ticker', 'Unknown')} - Change: {loser.get('change_percentage', 'N/A')} "
                f"(Price: {loser.get('price', 'N/A')}, Volume: {loser.get('volume', 'N/A')})"
                for loser in top_5_losers
            ]
        )

        summarized_tickers = summarize_content(
            raw_ticker_content, 
            "Summarize the following top 5 gainers and losers for a financial newsletter:"
        )

        # Combine summarized data for the newsletter
        combined_data = f"News Summaries:\n{summarized_news}\n\nTicker Trends:\n{summarized_tickers}"

        # Generate the final newsletter
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial newsletter editor."},
                {"role": "user", "content": f"Generate a newsletter using the following summarized data:\n{combined_data}"}
            ]
        )
        newsletter = response.choices[0].message.content

        # Display the newsletter
        st.subheader("Generated Newsletter")
        st.write(newsletter)
    except Exception as e:
        st.error(f"Error generating newsletter: {e}")

# Main Logic
if option == "Load News Data":
    load_news_data()
elif option == "Retrieve News Data":
    retrieve_news_data()
elif option == "Load Ticker Trends Data":
    load_ticker_trends_data()
elif option == "Retrieve Ticker Trends Data":
    retrieve_ticker_trends_data()
elif option == "Generate Newsletter":
    generate_newsletter()
