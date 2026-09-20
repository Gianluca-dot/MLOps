import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_sentiment_model(model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """Carica il tokenizer e il modello pre-addestrato."""
    print(f"🤖 Caricamento del modello {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = load_sentiment_model()
    print("✅ Modello e Tokenizer caricati correttamente.")
