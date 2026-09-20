import pandas as pd
from datasets import load_dataset

def load_data():
    """Carica il dataset di sentiment analysis (es. Twitter RoBERTa sentiment)."""
    print("📥 Caricamento del dataset in corso...")
    # Dataset di esempio per sentiment monitoring
    dataset = load_dataset("cardiffnlp/twitter_sentiment_multilingual", "en", split="train[:100]")
    df = pd.DataFrame(dataset)
    return df

if __name__ == "__main__":
    df = load_data()
    print(f"✅ Dataset caricato con successo. Righe: {len(df)}")
