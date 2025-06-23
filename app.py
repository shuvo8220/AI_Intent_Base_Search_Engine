# streamlit_app.py
import streamlit as st
import faiss
import numpy as np
import requests
import os
import json
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
import ollama

# ---- Config ----
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)
product_data = []
bert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

DEEPSEEK_70B_API_KEY = os.getenv("DEEPSEEK_70B_API_KEY")
DEEPSEEK_70B_API_URL = "https://api.deepseek.com/v1/chat/completions"

# ---- UI Styling ----
st.set_page_config(page_title="AI Product Search", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f0f2f6; padding: 20px;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px;}
    .stTextInput>div>input {border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# ---- Helper Functions ----
def encode_text(text):
    return bert_model.encode([text])[0]

def input_guardrails(text):
    return (len(text.strip()) >= 5, "Query is too short. Please provide more details.")

def output_guardrails(text):
    return (len(text) > 10 and "error" not in text.lower(), "Unreliable model output")

# ---- Prompt Templates ----
def build_ollama_prompt(query, local_results):
    results_text = "\n".join([f"- {r['name']}: {r['description']}" for r in local_results]) or "No local products available."
    return f"""
You are a smart product analysis assistant. Based on the following question and product list, generate a helpful response.

### User Query:
{query}

### Local Product Descriptions:
{results_text}

### Instructions:
1. List potential product options (if applicable).
2. Provide product pricing estimates (if any).
3. Highlight key features and benefits.
4. Compare products with pros/cons.
5. Add 3-day trend forecast (if relevant).

Respond in a clear, structured format using bullet points.
"""

def build_deepseek_prompt(query, local_results, external_results):
    local_text = "\n".join([f"- {r['name']}: {r['description']}" for r in local_results]) or "No local product matches."
    external_text = "\n".join([f"- {r['title']} ({r['href']})" for r in external_results]) or "No external results found."
    return f"""
You are an advanced AI agent specializing in product research and comparison.

### User Question:
{query}

### Local Search Data:
{local_text}

### External Web Sources:
{external_text}

### Output Requirements:
- List top 5 recommended products (with reasoning).
- Estimate pricing and specifications.
- Share customer reviews/insights (if known).
- Predict upcoming trends or product updates over the next 3 days.

Use markdown and structured bullet points. Ensure clarity and helpfulness.
"""

# ---- AI Functions ----
def generate_response(query, local_results):
    prompt = build_ollama_prompt(query, local_results)
    try:
        response = ollama.chat(model="deepseek-coder:1.3b", messages=[{"role": "user", "content": prompt}])
        return response.get("message", {}).get("content", "AI Error")
    except Exception as e:
        return f"AI Error: {e}"

def analyze_with_deepseek_70b(query, local_results, external_results):
    prompt = build_deepseek_prompt(query, local_results, external_results)
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_70B_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-70b",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(DEEPSEEK_70B_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except Exception as e:
        return f"API Error: {e}"

def search_external(query):
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=5))
    except Exception as e:
        return [{"title": "External search failed!", "href": "#", "body": str(e)}]

# ---- Streamlit App ----
st.title("\U0001F50D AI Product Search Engine")
st.write("Find and compare products using smart AI-based suggestions and search results.")

query = st.text_input("Ask about a product or comparison:", placeholder="e.g. Compare iPhone 15 vs Galaxy S24")

col1, col2 = st.columns(2)
with col1:
    with st.form("add_product_form"):
        st.subheader("Add a New Product")
        name = st.text_input("Product Name")
        desc = st.text_area("Product Description")
        submit = st.form_submit_button("Add Product")
        if submit:
            if name and desc:
                vec = encode_text(desc)
                index.add(np.array([vec], dtype=np.float32))
                product_data.append({"name": name, "description": desc})
                st.success(f"✅ Product '{name}' added.")
            else:
                st.error("Please fill in both fields.")

with col2:
    st.subheader("Feedback")
    feedback_text = st.text_area("Let us know your feedback")
    if st.button("Submit Feedback"):
        st.info("Thank you for your feedback!")
        with open("feedback_log.txt", "a") as f:
            f.write(json.dumps({"feedback": feedback_text}) + "\n")

if query:
    valid, msg = input_guardrails(query)
    if not valid:
        st.error(msg)
    else:
        query_vec = encode_text(query)
        local_results = []
        if index.ntotal > 0:
            distances, indices = index.search(np.array([query_vec], dtype=np.float32), 5)
            local_results = [product_data[idx] for idx in indices[0] if idx < len(product_data)]

        external_results = search_external(query)
        ai_resp = generate_response(query, local_results)
        deepseek_resp = analyze_with_deepseek_70b(query, local_results, external_results)

        safe, msg = output_guardrails(deepseek_resp)
        if not safe:
            st.warning("⚠️ Response from DeepSeek 70B may be unreliable. Showing fallback AI output.")
            st.write(ai_resp)
        else:
            st.subheader("\U0001F4A1 DeepSeek AI Response")
            st.markdown(deepseek_resp)

        st.subheader("\U0001F50E External Sources")
        for r in external_results:
            st.markdown(f"**[{r['title']}]({r['href']})** - {r['body']}")

        st.subheader("\U0001F4CB Local Product Matches")
        for r in local_results:
            st.markdown(f"- **{r['name']}**: {r['description']}")
