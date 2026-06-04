import pandas as pd

def load_raw_data(file_path):
    df = pd.read_csv(file_path)
    return df

def create_rfm_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    max_date = df['TransactionStartTime'].max()
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (max_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': lambda x: x[x > 0].sum()
    })
    return rfm
