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
st.title("Alpha Vantage Multi-Agent System with RAG, Bespoke Labs, and Chatbot")

### Helper Functions ###
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
        retrieved_data = []  # Placeholder for ChromaDB or any RAG retrieval

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

### Main Buttons ###
if st.button("Generate Newsletter"):
    st.write("Generating newsletter...")
    news_insights = researcher.execute_task("Extract insights from news data")
    market_insights = market_analyst.execute_task("Analyze market trends")
    risk_insights = risk_analyst.execute_task("Identify risks", additional_data=[news_insights, market_insights])
    generate_sequential_newsletter(news_insights, market_insights, risk_insights)

st.subheader("Fact-Check with Bespoke Labs")

if st.button("Fact-Check Newsletter"):
    factcheck_result = factcheck_with_bespoke_from_newsletter()
    if factcheck_result:
        st.write("**Fact-Check Result**")
        st.write(f"Support Probability: {factcheck_result['support_prob']}")
        st.write(f"Details: {factcheck_result['details']}")
