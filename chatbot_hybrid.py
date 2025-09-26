# chatbot_hybrid.py
import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ------------------------------
# 1️⃣ Load your merged CSV
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("groundwater_2016-2025_merged.csv")  # merged 7-year file

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]

    # Normalize district and state names
    for col in ["DISTRICT", "STATE"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # Fill NaNs for numeric fields
    for col in ["RECHARGE", "AVAILABLE", "EXTRACTION", "STAGE (%)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Ensure YEAR column exists
    if "YEAR" in df.columns:
        df["YEAR"] = df["YEAR"].astype(str)
    else:
        df["YEAR"] = "NA"

    return df

# ------------------------------
# 2️⃣ Create embeddings + FAISS index
# ------------------------------
@st.cache_resource
def create_index(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = []

    for _, row in df.iterrows():
        # Include year in the embedding text
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
    except:
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
        return "Hello! I can give you groundwater info for any district."
    if "how are you" in q:
        return "I'm doing great—always ready to talk about groundwater."
    if "thank" in q:
        return "You're welcome! Glad I could help."
    return None

# ------------------------------
# 4️⃣ Answer generator
# ------------------------------
def answer(query, model, index, texts, df):
    # First check for greetings / basic chat
    basic = basic_chat(query)
    if basic:
        return basic

    q = query.upper()

    # --- 1. Try to detect year in query ---
    year_match = re.findall(r"(20\d{2}(?:-\d{2})?)", q)
    year_query = year_match[0] if year_match else None

    # --- 2. Try to detect district in query ---
    matched_district = None
    for d in df["DISTRICT"].unique():
        if d in q:
            matched_district = d
            break


    # --- 3. If district found ---
    if matched_district:
        df_district = df[df["DISTRICT"] == matched_district]

        if year_query:
            df_year = df_district[df_district["YEAR"].str.contains(year_query)]
            if not df_year.empty:
                row = df_year.iloc[0]
            else:
                row = df_district.iloc[0]
        else:
            row = df_district.sort_values("YEAR", ascending=False).iloc[0]

        return (f"Here’s what I found for {row['DISTRICT']} ({row['STATE']}):\n\n"
                f"- Year: {row.get('YEAR','NA')}\n"
                f"- Recharge: {row['RECHARGE']} MCM\n"
                f"- Available: {row['AVAILABLE']} MCM\n"
                f"- Extraction: {row['EXTRACTION']} MCM\n"
                f"- Stage: {row.get('STAGE (%)','NA')}% "
                f"({stage_category(row.get('STAGE (%)',0))})")

    # --- 4. Semantic search fallback ---
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k=1)
    row = df.iloc[I[0][0]]

    return (f"I couldn’t find an exact match, but the closest data I have is:\n\n"
            f"- District: {row['DISTRICT']} ({row['STATE']})\n"
            f"- Year: {row.get('YEAR','NA')}\n"
            f"- Recharge: {row['RECHARGE']} MCM\n"
            f"- Available: {row['AVAILABLE']} MCM\n"
            f"- Extraction: {row['EXTRACTION']} MCM\n"
            f"- Stage: {row.get('STAGE (%)','NA')}% "
            f"({stage_category(row.get('STAGE (%)',0))})")

# ------------------------------
# 5️⃣ Streamlit UI
# ------------------------------
st.title("Groundwater AI Chatbot (7-year data, AI/ML powered)")
st.write("Ask me about recharge, extraction, or stage for any district.\n"
         "You can also say hi, hello, or thank me.\n"
         "Try a query like: Guntur 2020-21")

# Load data & create embeddings
df = load_data()
model, index, texts, df = create_index(df)

# User input
user_query = st.text_input("You:", placeholder="e.g. How much recharge is in Guntur 2020-21?")
if st.button("Ask") and user_query.strip():
    result = answer(user_query, model, index, texts, df)
    st.success(result)
