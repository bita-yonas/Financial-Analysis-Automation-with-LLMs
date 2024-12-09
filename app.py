import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import yfinance as yf
import pandas as pd
from groq import Groq
import os

# Fetch API keys and environment from Streamlit secrets
pinecone_api_key = st.secrets["PINECONE"]["API_KEY"]
pinecone_env = st.secrets["PINECONE"]["ENVIRONMENT"]
groq_api_key = st.secrets["GROQ"]["API_KEY"]

# Initialize Groq
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set. Please check your Streamlit secrets.")
groq_client = Groq(api_key=groq_api_key)

# Initialize Pinecone
if not pinecone_api_key or not pinecone_env:
    raise ValueError("PINECONE API_KEY or ENVIRONMENT is not set. Please check your Streamlit secrets.")
pc = Pinecone(
    api_key=pinecone_api_key,
    environment=pinecone_env
)

# Set the correct index name
INDEX_NAME = "stocks"
namespace = "stock-descriptions"

if INDEX_NAME not in pc.list_indexes().names():
    st.error(f"Index '{INDEX_NAME}' not found. Please ensure the index exists in your Pinecone environment.")
else:
    index = pc.Index(INDEX_NAME)

# Load SentenceTransformer model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Function to query stocks from Pinecone
def query_pinecone(query, namespace="stock-descriptions", top_k=20):
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    return results.get("matches", [])

# Function to determine the color for "52 Week Change"
def get_52_week_color(value):
    if value == "N/A":
        return "#CCCCCC"  # Neutral gray for "N/A"
    try:
        value = float(value.strip('%'))
        if value > 0:
            return "#00FF00"  # Green for positive values
        elif value < 0:
            return "#FF0000"  # Red for negative values
    except ValueError:
        pass
    return "#FFFFFF"  # Default white for any other case

# Function to sort and filter stocks based on a chosen metric
def get_top_stocks(matches, metric="Earnings Growth", top_n=6):
    stocks = []
    seen_stocks = set()  # Track unique stock names to avoid duplicates
    for match in matches:
        metadata = match.get("metadata", {})
        stock_name = metadata.get("Name", None)
        if stock_name and stock_name not in seen_stocks:  # Avoid duplicates
            seen_stocks.add(stock_name)
            try:
                value = metadata.get(metric, "N/A")
                value = float(value.strip('%')) if value != "N/A" else None
            except ValueError:
                value = None
            stocks.append((metadata, value))
    sorted_stocks = sorted(stocks, key=lambda x: (x[1] is not None, x[1]), reverse=True)
    return [stock[0] for stock in sorted_stocks[:top_n]]  # Return top_n metadata

# Function to fetch stock price data
def fetch_stock_prices(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        normalized_data = data / data.iloc[0] * 100  # Normalize to start at 100%
        return normalized_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Function to generate stock comparison summary using Groq
def generate_groq_summary(stocks_metadata):
    context = ""
    for stock in stocks_metadata:
        name = stock.get('Name', 'Unknown Name')
        earnings_growth = stock.get('Earnings Growth', 'N/A')
        revenue_growth = stock.get('Revenue Growth', 'N/A')
        gross_margins = stock.get('Gross Margins', 'N/A')
        ebitda_margins = stock.get('EBITDA Margins', 'N/A')
        week_change = stock.get('52 Week Change', 'N/A')
        context += (
            f"Stock Name: {name}\n"
            f"Earnings Growth: {earnings_growth}%\n"
            f"Revenue Growth: {revenue_growth}%\n"
            f"Gross Margins: {gross_margins}%\n"
            f"EBITDA Margins: {ebitda_margins}%\n"
            f"52 Week Change: {week_change}%\n\n"
        )

    prompt = (
        "You are a financial analyst. Based on the following stock data, generate a professional stock comparison "
        "summary highlighting strengths, weaknesses, and unique features of each stock:\n\n"
        f"{context}\n"
        "Provide a clear and concise summary."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a financial analysis assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating Groq summary: {e}")
        return None

# UI Layout
st.set_page_config(page_title="Automated Stock Analysis", layout="wide")
st.title("Automated Stock Analysis")
st.markdown("Enter a description of the kinds of stocks you are looking for:")

# Input for user query
query = st.text_input("", placeholder="e.g., data center builders", label_visibility="collapsed")

# Metric selection for sorting
metric_options = ["Earnings Growth", "Revenue Growth", "Gross Margins", "EBITDA Margins", "52 Week Change"]
selected_metric = st.selectbox("Select a metric to rank the top stocks:", metric_options, index=0)

if st.button("Find Stocks"):
    found_stocks = []
    if query:
        with st.spinner("Searching for stocks..."):
            matches = query_pinecone(query)
            if matches:
                # Get the top stocks based on the selected metric
                top_stocks = get_top_stocks(matches, metric=selected_metric, top_n=6)
                st.write("## Top Relevant Stocks")
                cols = st.columns(3)  # Display 3 cards per row
                for i, metadata in enumerate(top_stocks):  # Limit to top 6 results
                    found_stocks.append(metadata.get("Ticker", "Unknown Ticker"))
                    with cols[i % 3]:  # Wrap around every 3 results
                        st.markdown(
                            f"""
                            <div style="background-color:#111; padding:20px; border-radius:10px; margin-bottom:20px; box-shadow: 0px 4px 10px rgba(0,0,0,0.2);">
                                <h3 style="color:#FFFFFF; margin-bottom:5px;">{metadata.get('Name', 'Unknown Name')}</h3>
                                <p style="color:#CCCCCC; font-size:14px; margin-bottom:10px;">
                                    {metadata.get('Business Summary', 'No description available.')[:150]}...
                                </p>
                                <a href="{metadata.get('Website', '#')}" style="color:#1f8ef1; font-weight:bold; text-decoration:none;" target="_blank">Visit Website</a>
                                <hr style="border:none; height:1px; background-color:#444; margin:10px 0;">
                                <div style="display:grid; grid-template-columns:repeat(2, 1fr); gap:10px;">
                                    <div>
                                        <p style="font-size:14px; color:#aaa; margin:0;">Earnings Growth:</p>
                                        <p style="color:#FFFFFF; font-size:16px; margin:0;">
                                            {metadata.get('Earnings Growth', 'N/A')}%
                                        </p>
                                    </div>
                                    <div>
                                        <p style="font-size:14px; color:#aaa; margin:0;">Revenue Growth:</p>
                                        <p style="color:#FFFFFF; font-size:16px; margin:0;">
                                            {metadata.get('Revenue Growth', 'N/A')}%
                                        </p>
                                    </div>
                                    <div>
                                        <p style="font-size:14px; color:#aaa; margin:0;">Gross Margins:</p>
                                        <p style="color:#FFFFFF; font-size:16px; margin:0;">
                                            {metadata.get('Gross Margins', 'N/A')}%
                                        </p>
                                    </div>
                                    <div>
                                        <p style="font-size:14px; color:#aaa; margin:0;">EBITDA Margins:</p>
                                        <p style="color:#FFFFFF; font-size:16px; margin:0;">
                                            {metadata.get('EBITDA Margins', 'N/A')}%
                                        </p>
                                    </div>
                                    <div>
                                        <p style="font-size:14px; color:#aaa; margin:0;">52 Week Change:</p>
                                        <p style="color:{get_52_week_color(metadata.get('52 Week Change', 'N/A'))}; font-size:16px; margin:0;">
                                            {metadata.get('52 Week Change', 'N/A')}%
                                        </p>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # Stock price comparison
                if found_stocks:
                    st.markdown("### Stock Price Comparison")

                    start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
                    end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

                    selected_stocks = st.multiselect(
                        "Select stocks to plot",
                        options=found_stocks,
                        default=found_stocks
                    )
                    stock_prices = fetch_stock_prices(selected_stocks, start_date, end_date)
                    if stock_prices is not None:
                        st.line_chart(stock_prices)
                        st.download_button(
                            label="Download data as CSV",
                            data=stock_prices.to_csv().encode("utf-8"),
                            file_name="stock_prices_comparison.csv",
                            mime="text/csv"
                        )

                # Stock comparison summary
                st.markdown("### Stock Comparison Summary")
                summary = generate_groq_summary(top_stocks)
                if summary:
                    st.markdown(summary)

            else:
                st.warning("No matching stocks found. Try another query.")
    else:
        st.error("Please enter a query.")
