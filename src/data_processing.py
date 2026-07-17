import pandas as pd
import numpy as np

def load_raw_data(file_path):
    """Load raw transaction data"""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    return df

def create_rfm_features(df):
    """Create RFM features"""
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    max_date = df['TransactionStartTime'].max()
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (max_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': lambda x: x[x > 0].sum()
    }).rename(columns={
        'TransactionStartTime': 'recency',
        'TransactionId': 'frequency',
        'Amount': 'monetary'
    })
    return rfm

def create_proxy_default(rfm_df, bad_ratio=0.3, good_ratio=0.2):
    """Create proxy default target"""
    rfm_df = rfm_df.copy()
    n_segments = 3
    
    rfm_df['r_score'] = pd.qcut(rfm_df['recency'].rank(method='first'), n_segments, labels=False, duplicates='drop')
    rfm_df['f_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), n_segments, labels=False, duplicates='drop')
    rfm_df['m_score'] = pd.qcut(rfm_df['monetary'].rank(method='first'), n_segments, labels=False, duplicates='drop')
    rfm_df['rfm_score'] = rfm_df['r_score'] + rfm_df['f_score'] + rfm_df['m_score']
    
    threshold_bad = rfm_df['rfm_score'].quantile(bad_ratio)
    threshold_good = rfm_df['rfm_score'].quantile(1 - good_ratio)
    
    rfm_df['default'] = 1
    rfm_df.loc[rfm_df['rfm_score'] >= threshold_good, 'default'] = 0
    
    return rfm_df
