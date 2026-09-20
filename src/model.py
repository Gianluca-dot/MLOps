import os
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentAnalyzer:
    """Classe wrapper per il caricamento del modello RoBERTa e l'esecuzione dell'inferenza."""

    def __init__(self, config_path: str = "config/config.yaml"):
        # Carica le configurazioni
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            model_name = self.config.get("model", {}).get(
                "name", "cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
        else:
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

        # Impostazione Device (GPU T4 su Colab se disponibile, altrimenti CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Mappatura etichette
        self.labels_map = {0: "negative", 1: "neutral", 2: "positive"}

        # Caricamento Tokenizer e Modello
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()

    def predict_single(self, text: str) -> dict:
        """Esegue l'inferenza su un singolo testo."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).squeeze(0)

        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

        return {
            "text": text,
            "label": self.labels_map.get(pred_idx, str(pred_idx)),
            "confidence": confidence,
            "scores": {
                self.labels_map[i]: probs[i].item() for i in range(len(probs))
            },
        }


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    res = analyzer.predict_single(
        "Great service and amazing experience with MachineInnovators!"
    )
    print("Test esecuzione diretta:", res)
