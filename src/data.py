import pandas as pd
from datasets import load_dataset


def load_data():
    """Carica un campione del dataset tweet_eval per la Sentiment Analysis."""
    print("📥 Caricamento del dataset in corso...")
    # Utilizza il path canonico 'cardiffnlp/tweet_eval' per la piena compatibilità con huggingface_hub
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train[:100]")
    df = pd.DataFrame(dataset)
    return df


if __name__ == "__main__":
    df = load_data()
    print(f"✅ Dati caricati con successo. Formato DataFrame: {df.shape}")
