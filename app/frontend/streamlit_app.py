from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import requests

st.set_page_config(page_title="Langagraph Agent UI", layout="centered")
st.title("AI Chatbot Agent")
st.write("Create and Interact with the AI Agent")

SYSTEM_PROMPT_DEFAULT = "Act as an AI chatbot who is smart and friendly."

system_prompt = st.text_area(
    "Define your AI Agent: ",
    height = 70,
    placeholder = "Type your system prompt here...",
    value=SYSTEM_PROMPT_DEFAULT,
)

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = st.radio("Select Provider:", {"Groq", "OpenAI"})

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

allow_web_search = st.checkbox("Allow Web Search")

def should_use_web_search(query: str) -> bool:
    q = query.lower()
    # basic rule to not do web search for personal queries
    personal_starters = ["should i", "do you think", "what should i do", "how do i deal with"]
    if any(q.startswith(p) for p in personal_starters):
        return False
    return True

user_query = st.text_area(
    "Enter your query: ",
    height=150,
    placeholder="Type your system prompt here..."
)

API_URL = "http://localhost:9999/chat"

if st.button("Ask Agent!"):
    if user_query.strip():

        allow_search_final = allow_web_search and should_use_web_search(user_query)

        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
        except requests.RequestException as e:
            st.error(f"Failed to reach backend: {e}")
        else:
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                st.error(f"Backend error: {detail}")
            else:
                data = resp.json()
                answer = data.get("answer") or data
                st.subheader("Agent Response")
                st.markdown(f"{answer}")

    else:
        st.warning("Please enter a query first")