import pandas as pd
from datasets import load_dataset
import os

def load_and_prep_data(data_path="data/raw_feedback.csv"):
    """Fetches Yelp reviews from Hugging Face or loads from local CSV."""
    # Load from local cache if it exists
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    
    # Fallback: Fetch from Hugging Face
    dataset = load_dataset("yelp_review_full", split="train[:500]") 
    df = pd.DataFrame(dataset)
    df['stars'] = df['label'] + 1
    df = df[df['stars'] < 3].copy().reset_index(drop=True)
    df = df.rename(columns={'text': 'document'})
    df = df[['document', 'stars']]
    
    # Ensure directory exists and save locally
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df.to_csv(data_path, index=False)
    
    return df