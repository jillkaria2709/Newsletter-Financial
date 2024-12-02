# Financial Newsletter Generation using Multi-Agent System

This project is a **Streamlit-based application** that generates professional financial newsletters using a multi-agent system (MAS) architecture. The project integrates **ChromaDB**, **OpenAI GPT models**, and **Alpha Vantage APIs** to fetch, process, and analyze financial data. The generated newsletters provide valuable insights into market trends, news sentiment, and risk factors, making it a comprehensive tool for financial professionals.

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Architecture](#architecture)  
4. [Prerequisites](#prerequisites)  
5. [Step-by-Step Guide](#step-by-step-guide)  
6. [How It Works](#how-it-works)  
7. [Key Components](#key-components)  
8. [Future Enhancements](#future-enhancements)

## Overview

This application combines the capabilities of:  
- **Multi-Agent Systems**: Agents like `Researcher`, `Market Analyst`, `Risk Analyst`, and `Writer` perform specific roles.  
- **ChromaDB**: Used for **Retrieval-Augmented Generation (RAG)** to fetch relevant financial data from structured collections.  
- **OpenAI GPT-4 API**: For advanced natural language processing tasks.  
- **Alpha Vantage API**: To retrieve stock and market data.  

The application is built with a modular and scalable architecture to enable seamless data retrieval, processing, and newsletter generation.

## Features

- **Data Fetching and Processing**:  
  - Fetch financial news and market trends via Alpha Vantage.  
  - Use ChromaDB for structured storage and retrieval of relevant financial data.  

- **Multi-Agent Task Execution**:  
  - Extract insights from data using agents (`Researcher`, `Market Analyst`, `Risk Analyst`, `Writer`).  

- **Newsletter Generation**:  
  - Generate a professional newsletter based on agent insights.  
  - Use financial jargon and concise reporting.  

- **Interactive Chatbot**:  
  - Answer user queries with either RAG or OpenAI GPT fallback.  

- **CSV Upload**:  
  - Analyze investor portfolios and fetch real-time stock data.  

- **Fact-Checking**:  
  - Verify newsletter content using Bespoke Labs.

## Architecture

The project utilizes a **multi-agent system** with the following agents:  
1. **Researcher**: Extracts news insights.  
2. **Market Analyst**: Analyzes market trends.  
3. **Risk Analyst**: Identifies financial risks.  
4. **Writer**: Generates professional newsletters.  

The backend is supported by:  
- **ChromaDB**: Persistent client for storing and retrieving structured data.  
- **OpenAI GPT**: For natural language generation.  
- **Alpha Vantage API**: For real-time market data.

## Prerequisites

1. Python 3.8 or later  
2. A **Streamlit Cloud** or local Streamlit setup  
3. API keys:  
   - **Alpha Vantage API**  
   - **OpenAI GPT API**  
   - **Bespoke Labs API**  
4. Libraries: See [requirements.txt](requirements.txt)

## Step-by-Step Guide

### Step 1: Update Data
Use the **Update News Data** and **Update Market Data** buttons to fetch the latest information from Alpha Vantage. These actions populate ChromaDB collections (`news_sentiment_data` and `market_data`).

### Step 2: Generate a Newsletter
Click **Generate Newsletter** to process insights from the `Researcher`, `Market Analyst`, and `Risk Analyst`. The **Writer agent** compiles the insights into a professional financial newsletter.

### Step 3: Fact-Check
Use the **Fact-Check Newsletter** button to verify the generated content using Bespoke Labs.

### Step 4: Analyze Portfolio
Upload a CSV file containing investor data (columns: Investor Name, Company, Invested Amount, Returns, Returns %, Ticker). Fetch stock data and generate investment recommendations using OpenAI.

### Step 5: Use Chatbot
Enter a query in the chatbot interface. Queries are processed using tool-based handlers for stock ticker queries or RAG/OpenAI fallback for general questions.

## How It Works

### Data Retrieval
1. Fetch data using Alpha Vantage APIs:  
   - **News Sentiment API**: Retrieves the latest news.  
   - **Market Data API**: Retrieves top gainers, losers, and actively traded stocks.  

2. Update ChromaDB collections for efficient retrieval.

### Multi-Agent Execution
Agents execute specific tasks based on user inputs or predefined goals. Insights are compiled to form a comprehensive context for OpenAI GPT-4.

### Newsletter Generation
Insights are processed into a structured newsletter with sections:
1. **Introduction**  
2. **News Insights**  
3. **Market Trends**  
4. **Risk Assessments**  
5. **Conclusion**

### Fact-Checking
The newsletter is validated using Bespoke Labs' fact-checking feature.

### Portfolio Analysis
Fetch real-time stock data for uploaded CSV files and generate investment recommendations using OpenAI GPT-4.

## Key Components

### Helper Functions
- **`fetch_ticker_price`**: Retrieves stock prices using Alpha Vantage.  
- **`retrieve_from_rag`**: Fetches data from ChromaDB collections.  
- **`call_openai_gpt4`**: Interacts with OpenAI GPT models.

### Agents
- Modular agent design allows for easy scalability and addition of new roles.

### Fact-Checking
- Integration with Bespoke Labs ensures the accuracy of generated content.

## Future Enhancements

- Add more sophisticated market analysis features.  
- Expand support for multiple data sources (e.g., Bloomberg, Reuters).  
- Enhance the chatbot with voice input capabilities.  
- Optimize ChromaDB queries for better performance.
