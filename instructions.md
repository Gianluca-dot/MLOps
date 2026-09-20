# ⚠️ Instructions & Warnings

## ⚠️ Warning Importante: Disattivare la Traduzione Automatica
Se utilizzi il browser (es. Google Chrome) con la traduzione automatica attiva, **disattivala** quando navighi o crei i file di questo progetto su GitHub. La traduzione automatica altera i nomi dei file (es. traducendo `data.py` in `dati.py` o `train.py` in `allenarsi.py`), rompendo completamente i test di pytest, le importazioni Python e la pipeline CI/CD.

---

## 💻 Comandi Principali e Guide Rapide

### 1. Esecuzione dei Test Locali (con Pytest)
Per verificare che tutti i moduli (`src/`) e i test (`tests/`) funzionino correttamente:
```bash

# Dipendenze
pip install -r requirements.txt

# Test unitari
pytest -v

# Valutazione e metriche
python src/evaluate.py

# Training / Retraining
python src/train.py

# Dashboard Streamlit
streamlit run app.py
