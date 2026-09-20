import pandas as pd
from src.data import load_data

def test_load_data():
    """Verifica che la funzione load_data restituisca un DataFrame non vuoto."""
    df = load_data()
    assert isinstance(df, pd.DataFrame), "Il risultato deve essere un DataFrame pandas."
    assert not df.empty, "Il dataset caricato non deve essere vuoto."
    assert len(df) > 0, "Il dataset deve contenere almeno una riga."
