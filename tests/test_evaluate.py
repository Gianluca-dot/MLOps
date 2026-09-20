import os
from src.evaluate import run_evaluation

def test_run_evaluation():
    """Verifica che la valutazione restituisca un dizionario con le metriche attese."""
    metrics = run_evaluation()
    
    assert isinstance(metrics, dict), "Le metriche restituite devono essere in formato dizionario."
    assert "accuracy" in metrics, "Il dizionario deve contenere la metrica 'accuracy'."
    assert "f1_macro" in metrics, "Il dizionario deve contenere la metrica 'f1_macro'."
    
    # Verifica che i valori siano numerici e validi (tra 0 e 1)
    assert 0.0 <= metrics["accuracy"] <= 1.0, "L'accuracy deve essere compresa tra 0 e 1."
    assert 0.0 <= metrics["f1_macro"] <= 1.0, "L'f1_macro deve essere compresa tra 0 e 1."
