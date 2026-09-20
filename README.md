# 📊 MLOps

Repository per il monitoraggio continuo, test e deployment di un modello di Sentiment Analysis basato su Hugging Face Transformers (`cardiffnlp/twitter-roberta-base-sentiment-latest`).

---

## ⚠️ WARNING: Disattivare la Traduzione Automatica
**Attenzione:** Disattiva qualsiasi estensione o traduttore automatico del browser (come Google Translate) mentre lavori su GitHub per questo progetto. La traduzione automatica modifica i nomi dei file di sistema (es. traducendo `data.py` in file in italiano), generando errori bloccanti nei test e nella pipeline CI/CD.

---

## 🛠️ Sintesi della Costruzione del Progetto
Questo repository è stato strutturato azzerando il precedente disordine e adottando un approccio modulare e pulito:
1. **Core Modulare (`src/`)**: Separazione netta tra caricamento dati (`data.py`), gestione del modello Hugging Face (`model.py`), calcolo metriche (`evaluate.py`) e logica di training (`train.py`).
2. **Testing Automatismi (`tests/`)**: Suite di test unitari con `pytest` per validare robustezza e integrità ad ogni modifica.
3. **CI/CD & Monitoring**: Pipeline GitHub Actions (`.github/workflows/ci.yml`) per i test automatici e dashboard interattiva in Streamlit (`app.py`) per il monitoraggio visivo delle metriche.
4. **Google Colab (`notebooks/`)**: Sganciato dal codice monolitico e configurato per clonare questa repository pulita, eseguendo il fine-tuning su GPU in modo snello.

---

## 📁 Struttura della Repository

```text
MLOps/
│
├── .github/
│   workflows/
│       └── ci.yml             # Pipeline CI/CD per l'esecuzione automatica dei test
├── config/
│   └── config.yaml            # File di configurazione dei parametri di training
├── data/
│   └── metrics.json           # Metriche di performance (Accuracy, F1 Macro, F1 Weighted)
├── notebooks/
│   └── retrain_pipeline.ipynb # Notebook Colab per fine-tuning end-to-end su GPU
├── src/
│   ├── __init__.py
│   ├── data.py                # Gestione e caricamento del dataset
│   ├── evaluate.py            # Script di valutazione e lettura metriche
│   ├── model.py               # Caricamento del modello e tokenizer Hugging Face
│   └── train.py               # Logica di training e aggiornamento metriche
├── tests/
│   ├── __init__.py
│   ├── test_data.py           # Test unitari per i dati
│   ├── test_evaluate.py       # Test unitari per le metriche
│   └── test_model.py          # Test unitari per il modello
├── .gitignore                 # File e cartelle ignorate da Git
├── app.py                     # Dashboard interattiva in Streamlit
├── instructions.md            # Guida comandi e avvisi importanti
└── requirements.txt           # Dipendenze del progetto
