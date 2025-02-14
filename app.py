import streamlit as st
import pandas as pd
import docx
import openai
import google.generativeai as genai
import requests
from textblob import TextBlob
import io
import nltk
import plotly.graph_objects as go
import plotly.express as px
import re

# Initialize NLTK
@st.cache_resource
def initialize_nltk():
    nltk.download('punkt')
    return True

initialize_nltk()

# OpenAI API Function
def process_with_openai(prompt, text, task):
    try:
        client = openai.OpenAI(api_key=st.session_state.api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a qualitative analysis expert."},
                {"role": "user", "content": f"Task: {task}\n\nQuestion: {prompt}\n\nText to analyze: {text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with OpenAI API: {str(e)}"

# Google Gemini API Function
def process_with_gemini(prompt, text, task):
    try:
        genai.configure(api_key=st.session_state.api_key)
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(f"Task: {task}\n\nQuestion: {prompt}\n\nText to analyze: {text}")
        return response.text
    except Exception as e:
        return f"Error with Google Gemini API: {str(e)}"

# Deepseek API Function
def process_with_deepseek(prompt, text, task):
    try:
        headers = {
            "Authorization": f"Bearer {st.session_state.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": [
                {"role": "user", "content": f"Task: {task}\n\nQuestion: {prompt}\n\nText to analyze: {text}"}
            ],
            "max_tokens": 500,
            "stream": False
        }
        response = requests.post("https://router.huggingface.co/together/v1/chat/completions", json=payload, headers=headers)
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except Exception as e:
        return f"Error with Deepseek API: {str(e)}"

# Main UI Setup
st.title("Qualitative Analysis Chatbot")

# Sidebar Configuration
with st.sidebar:
    st.title("Configuration")
    api_choice = st.radio("Select AI Model:", ["OpenAI", "Google Gemini", "Deepseek-R1"])
    api_key = st.text_input("Enter API Key:", type="password")
    if st.button("Save Configuration"):
        st.session_state.api_choice = api_choice
        st.session_state.api_key = api_key
        st.success(f"Configuration saved for {api_choice}")

# Input Handling
input_method = st.radio("Choose input method:", ["Text Input", "File Upload"])
user_question = st.text_area("Enter your question:", height=100)
qualitative_data = ""
if input_method == "Text Input":
    qualitative_data = st.text_area("Paste your qualitative data here:", height=200)
else:
    uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx', 'docx'])
    if uploaded_file:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            qualitative_data = df.to_string()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            qualitative_data = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        st.success("File content loaded successfully!")

# Perform Analysis
def create_sentiment_plots(sentences):
    sentiment_scores = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
    sentiment_labels = ["Very Negative" if s <= -0.6 else "Negative" if -0.6 < s <= -0.2 else "Neutral" if -0.2 < s < 0.2 else "Positive" if 0.2 <= s < 0.6 else "Very Positive" for s in sentiment_scores]
    sentiment_distribution = {label: sentiment_labels.count(label) for label in set(sentiment_labels)}
    
    # Bar Chart
    bar_fig = go.Figure([go.Bar(x=list(sentiment_distribution.keys()), y=list(sentiment_distribution.values()), marker_color=['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#0074D9'])])
    bar_fig.update_layout(title='Sentiment Distribution', xaxis_title='Sentiment Category', yaxis_title='Count', template='plotly_white')
    
    # Sentiment Flow Chart
    flow_fig = go.Figure([go.Scatter(y=sentiment_scores, mode='lines+markers', marker=dict(size=8), line=dict(width=2))])
    flow_fig.update_layout(title='Sentiment Flow', xaxis_title='Sentence Number', yaxis_title='Sentiment Score', yaxis=dict(range=[-1, 1]), template='plotly_white')
    
    return bar_fig, flow_fig

if st.button("Analyze"):
    if not st.session_state.api_key:
        st.error("Please configure your API key in the sidebar first.")
    elif not user_question or not qualitative_data:
        st.error("Please provide both a question and data to analyze.")
    else:
        with st.spinner("Analyzing your data..."):
            blob = TextBlob(qualitative_data)
            overall_sentiment = blob.sentiment
            sentences = qualitative_data.split('.')
            
            st.metric("Overall Sentiment Score", f"{overall_sentiment.polarity:.2f}")
            st.metric("Overall Subjectivity", f"{overall_sentiment.subjectivity:.2f}")
            
            bar_fig, flow_fig = create_sentiment_plots(sentences)
            st.plotly_chart(bar_fig)
            st.plotly_chart(flow_fig)
            
            st.write("### AI Analysis")
            if st.session_state.api_choice == "OpenAI":
                summary = process_with_openai(user_question, qualitative_data, "Summarize the key themes and main points from the text.")
            elif st.session_state.api_choice == "Google Gemini":
                summary = process_with_gemini(user_question, qualitative_data, "Summarize the key themes and main points from the text.")
            else:
                st.write("Thinking...")
                summary = process_with_deepseek(user_question, qualitative_data, "Summarize the key themes and main points from the text.")
            
            st.write("**Summary:**")
            st.write(summary)
