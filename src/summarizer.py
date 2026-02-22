from transformers import pipeline
import logging

class ClusterSummarizer:
    def __init__(self):
        # Using a lightweight local summarizer
        try:
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")
        except Exception as e:
            logging.warning(f"Summarizer failed to initialize (check if torch is installed): {e}")
            self.summarizer = None

    def summarize(self, cluster_texts):
        """Generate a label for the cluster using a local LLM."""
        if self.summarizer is None:
            return "Summary Unavailable"
            
        context = " ".join(cluster_texts[:5]) 
        context = context[:1000] # Truncate for token limits
        
        if len(context.strip()) < 50:
            return "Miscellaneous Issues"
            
        summary = self.summarizer(context, max_length=20, min_length=5, do_sample=False)
        return summary[0]['summary_text']