import streamlit as st
import pandas as pd
import plotly.express as px
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
    
    tab_kmeans, tab_hdbscan = st.tabs(["K-Means Clustering", "HDBSCAN (Density)"])
    
    # Helper to render results
    def render_cluster_view(algo_name, **kwargs):
        with st.spinner(f"Running {algo_name}..."):
            labels, coords = discover_clusters(embeddings, algorithm=algo_name.lower(), **kwargs)
            
            # Update DataFrame for plotting
            df['cluster'] = labels
            df['PC1'] = coords[:, 0]
            df['PC2'] = coords[:, 1]
            
            # Visualization
            fig = px.scatter(df, x='PC1', y='PC2', color=df['cluster'].astype(str),
                             hover_data=['document'], 
                             title=f"{algo_name} Clusters (PCA Projection)",
                             labels={'PC1': 'Theme Dimension 1', 'PC2': 'Theme Dimension 2'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Summaries
            unique_labels = sorted(set(labels))
            for label in unique_labels:
                if label == -1:
                    cluster_label = "Noise / Outliers"
                else:
                    cluster_data = df[df['cluster'] == label]
                    cluster_texts = cluster_data['document'].tolist()
                    cluster_label = summarizer.summarize(cluster_texts)
                
                count = len(df[df['cluster'] == label])
                with st.expander(f"Cluster {label} | {cluster_label} ({count} records)"):
                    st.write("**Sample feedback:**")
                    for text in df[df['cluster'] == label]['document'].head(3):
                        st.markdown(f"- *{text}*")

    with tab_kmeans:
        k = st.slider("Number of Clusters", 3, 10, 5, key="k_slider")
        render_cluster_view("KMeans", num_clusters=k)
        
    with tab_hdbscan:
        min_size = st.slider("Minimum Cluster Size", 3, 20, 5, key="hdb_slider")
        st.caption("HDBSCAN automatically detects cluster counts and identifies noise (Label -1).")
        render_cluster_view("HDBSCAN", min_cluster_size=min_size)