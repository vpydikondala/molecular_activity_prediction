import pandas as pd

def prepare_data(filepath):
    """
    Load and preprocess the fingerprint dataset.
    """
    df = pd.read_csv(filepath)
    df.dropna(inplace=True)
    return df
