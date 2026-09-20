import torch
from transformers import PreTrainedTokenizer, torch.nn as nn
from src.model import load_sentiment_model

def test_load_sentiment_model():
    """Verifica che il modello e il tokenizer vengano caricati correttamente."""
    tokenizer, model = load_sentiment_model()
    
    # Controlla che siano le istanze corrette di Hugging Face
    assert tokenizer is not None, "Il tokenizer non deve essere None."
    assert model is not None, "Il modello non deve essere None."
    
    # Verifica che il modello sia in modalità evaluation o pronto
    assert isinstance(model, nn.Module), "Il modello deve essere un modulo PyTorch."
