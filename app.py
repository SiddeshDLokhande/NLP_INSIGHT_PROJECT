import streamlit as st
import pandas as pd
from src.data_ingestion import load_and_prep_data
from src.embedder import VectorStore
from src.cluster_engine import discover_clusters
from src.summarizer import ClusterSummarizer

# --- SETUP ---
st.set_page_config(page_title="Semantic Insight Engine", layout="wide")
st.title("Semantic Insight Engine")
st.write("Extracting actionable business intelligence from unstructured text feedback.")

@st.cache_resource
def initialize_components():
    """Load heavy models and DB connections once."""
    vector_store = VectorStore()
    summarizer = ClusterSummarizer()
    return vector_store, summarizer

@st.cache_data
def get_data():
    return load_and_prep_data()

# Initialize Backend
df = get_data()
vector_store, summarizer = initialize_components()

# Build or Load Persistent DB
with st.spinner("Connecting to Vector Database..."):
    collection, embeddings = vector_store.add_documents(df['document'].tolist())

# --- UI CONTROLS ---
st.sidebar.header("Pipeline Controls")
num_clusters = st.sidebar.slider("Number of Issue Clusters", min_value=3, max_value=10, value=5)

st.sidebar.markdown("---")
st.sidebar.write("**Semantic Search**")
search_query = st.sidebar.text_input("Query the database (e.g., 'terrible customer service')")

# --- MAIN LOGIC ---
if search_query:
    st.subheader(f"Semantic Search Results for: '{search_query}'")
    results = vector_store.query(search_query)
    
    if results['documents']:
        for doc in results['documents'][0]:
            st.info(doc)
else:
    st.subheader("Automated Theme Discovery")
    with st.spinner("Clustering embeddings and generating AI summaries..."):
        df['cluster'] = discover_clusters(embeddings, num_clusters)
        
        for i in range(num_clusters):
            cluster_data = df[df['cluster'] == i]
            cluster_texts = cluster_data['document'].tolist()
            
            cluster_label = summarizer.summarize(cluster_texts)
            
            with st.expander(f"Cluster {i+1} | AI Label: {cluster_label} ({len(cluster_data)} records)"):
                st.write("**Sample raw feedback driving this theme:**")
                for text in cluster_texts[:3]:
                    st.markdown(f"- *{text}*")