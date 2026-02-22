from transformers import pipeline

class ClusterSummarizer:
    def __init__(self):
        # Using a lightweight local summarizer
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

    def summarize(self, cluster_texts):
        """Generate a label for the cluster using a local LLM."""
        context = " ".join(cluster_texts[:5]) 
        context = context[:1000] # Truncate for token limits
        
        if len(context.strip()) < 50:
            return "Miscellaneous Issues"
            
        summary = self.summarizer(context, max_length=20, min_length=5, do_sample=False)
        return summary[0]['summary_text']