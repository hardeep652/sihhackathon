import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Set the page configuration for a centered layout
st.set_page_config(
    page_title="Groundwater Chatbot", 
    page_icon="💧", 
    layout="wide"
)

# Custom CSS for the page background and animations
st.markdown("""
<style>
    /* Professional Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #1c2541, #0b132b);
        color: white;
    }

    /* Professional Card-like Container for content */
    .content-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* Add a subtle animation to the titles */
    h1, h2, h3 {
        animation: fadeIn 1s ease-in-out;
    }

    /* Add a pulse animation to the button */
    .stButton > button {
        animation: pulse 2s infinite;
        background-color: #3a86ff;
        border: none;
        color: white;
    }
    
    .stButton > button:hover {
        background-color: #4b94ff;
    }

    /* Chat bubble styles */
    .st-chat-message-user .stChatMessageContent {
        background-color: #3a86ff;
        color: white;
        border-bottom-right-radius: 0;
    }
    .st-chat-message-assistant .stChatMessageContent {
        background-color: #4a4a4a;
        color: white;
        border-bottom-left-radius: 0;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# 1️⃣ Load your merged CSV
# ------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("groundwater_2016-2025_merged.csv")
        df.columns = [c.strip().upper() for c in df.columns]

        for col in ["DISTRICT", "STATE"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()

        for col in ["RECHARGE", "AVAILABLE", "EXTRACTION", "STAGE (%)"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if "YEAR" in df.columns:
            df["YEAR"] = df["YEAR"].astype(str)
        else:
            df["YEAR"] = "NA"
        return df
    except FileNotFoundError:
        st.error("Error: 'groundwater_2016-2025_merged.csv' not found. Please make sure the file is in the same directory.")
        return pd.DataFrame()

# ------------------------------
# 2️⃣ Create embeddings + FAISS index
# ------------------------------
@st.cache_resource
def create_index(df):
    if df.empty:
        return None, None, [], pd.DataFrame()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = []

    for _, row in df.iterrows():
        desc = (f"{row['DISTRICT']}, {row['STATE']}, Year {row.get('YEAR','NA')}: "
                f"Recharge {row['RECHARGE']}, Available {row['AVAILABLE']}, "
                f"Extraction {row['EXTRACTION']}, Stage {row.get('STAGE (%)','NA')}")
        texts.append(desc)

    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index, texts, df

# ------------------------------
# Helper: classify stage %
# ------------------------------
def stage_category(stage):
    try:
        stage = float(stage)
    except (ValueError, TypeError):
        return "Unknown"
    if stage < 70:
        return "Safe"
    elif stage < 90:
        return "Semi-Critical"
    elif stage < 100:
        return "Critical"
    else:
        return "Over-Exploited"

# ------------------------------
# 3️⃣ Basic conversational AI fallback
# ------------------------------
def basic_chat(query):
    q = query.lower()
    if any(greet in q for greet in ["hi", "hello", "hey"]):
        return "👋 Hello! I can give you groundwater info for any district."
    if "how are you" in q:
        return "💧 I'm flowing great—always ready to talk about groundwater."
    if "thank" in q:
        return "🙏 You're welcome! Glad I could help."
    return None

# ------------------------------
# 4️⃣ Answer generator
# ------------------------------
def answer(query, model, index, texts, df):
    if df.empty:
        return "Sorry, I can't access the data right now. Please check if the CSV file is in the correct location."
    
    basic = basic_chat(query)
    if basic:
        return basic

    q = query.upper()
    year_match = re.findall(r"(20\d{2})", q)
    year_query = year_match[0] if year_match else None

    matched_district = None
    for d in df["DISTRICT"].unique():
        if d in q:
            matched_district = d
            break

    if matched_district:
        df_district = df[df["DISTRICT"] == matched_district]
        if year_query:
            df_year = df_district[df_district["YEAR"].str.contains(year_query)]
            row = df_year.iloc[0] if not df_year.empty else df_district.iloc[0]
        else:
            row = df_district.sort_values("YEAR", ascending=False).iloc[0]

        return (f"📍 **{row['DISTRICT']} ({row['STATE']})**\n\n"
                f"📅 Year: {row.get('YEAR','NA')}  \n"
                f"💧 Recharge: {row['RECHARGE']} MCM  \n"
                f"💦 Available: {row['AVAILABLE']} MCM  \n"
                f"⚡ Extraction: {row['EXTRACTION']} MCM  \n"
                f"🚨 Stage: {row.get('STAGE (%)','NA')}% "
                f"({stage_category(row.get('STAGE (%)',0))})")

    query_vec = model.encode([query])
    D, I = index.search(query_vec, k=1)
    row = df.iloc[I[0][0]]

    return (f"🔎 Closest data I found:\n\n"
            f"📍 **{row['DISTRICT']} ({row['STATE']})**\n"
            f"📅 Year: {row.get('YEAR','NA')}  \n"
            f"💧 Recharge: {row['RECHARGE']} MCM  \n"
            f"💦 Available: {row['AVAILABLE']} MCM  \n"
            f"⚡ Extraction: {row['EXTRACTION']} MCM  \n"
            f"🚨 Stage: {row.get('STAGE (%)','NA')}% "
            f"({stage_category(row.get('STAGE (%)',0))})")

# ------------------------------
# 5️⃣ Main UI Logic
# ------------------------------
# Initialize session state for chat visibility and messages
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display either the landing page or the chat UI based on session state
if not st.session_state.chat_open:
    # --- Landing Page with Content Card ---
    # We combine the open and close markdown tags into a single block
    st.markdown(
        f"""
        <div class="content-card">
            <h1>🚀 Team Vyoma</h1>
            <h2>Problem Statement ID: 25066</h2>
            <h3>Development of an AI-driven ChatBOT for INGRES as a Virtual Assistant</h3>
            <hr>
            <p><h3>Background</h3>
            The Assessment of Dynamic Ground Water Resources of India is conducted annually by the Central Ground Water Board (CGWB) and State/UT Ground Water Departments...</p>
            <p><h3>Proposed Solution</h3>
            To improve accessibility, it is proposed to develop an AI-driven ChatBOT for INGRES.
            This intelligent virtual assistant will enable users to easily query groundwater data, access historical and current assessments, and obtain instant insights.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Button to open the chat
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💬 Chat with AI", key="open_chat"):
        st.session_state.chat_open = True
        st.rerun()

else:
    # --- Chatbot UI ---
    st.title("💧 Groundwater AI Chatbot")
    
    # Back button to close the chat
    if st.button("⬅ Back to Problem Statement", key="close_chat"):
        st.session_state.chat_open = False
        st.rerun()

    # Display chat history
    for role, msg in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(msg)

    # Get data and models
    df = load_data()
    model, index, texts, df_data = create_index(df)

    # Chat input box
    if user_query := st.chat_input("Type your question..."):
        # User message
        st.session_state.messages.append(("user", user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        # Bot response
        if df_data.empty:
            result = "I am having trouble accessing the data. Please check the file path."
        elif model and index:
            result = answer(user_query, model, index, texts, df_data)
        else:
            result = "I am having trouble loading the AI model. Please try again later."
            
        st.session_state.messages.append(("assistant", result))
        with st.chat_message("assistant"):
            st.markdown(result)