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

"""
Data processing module for credit risk modeling
Handles data loading, RFM feature engineering, and proxy target creation
"""

import pandas as pd
import numpy as np

def load_raw_data(file_path):
    """
    Load raw transaction data from CSV file
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading file: {e}")
        raise

def create_rfm_features(df):
    """
    Create RFM (Recency, Frequency, Monetary) features from transaction data
    
    Parameters:
    df (pd.DataFrame): Raw transaction data
    
    Returns:
    pd.DataFrame: Customer-level RFM features
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Convert timestamp to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Get latest transaction date for recency calculation
    max_date = df['TransactionStartTime'].max()
    
    # Group by CustomerId to create RFM metrics
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (max_date - x.max()).days,  # Recency
        'TransactionId': 'count',  # Frequency
        'Amount': lambda x: x[x > 0].sum()  # Monetary (positive amounts only)
    }).rename(columns={
        'TransactionStartTime': 'recency',
        'TransactionId': 'frequency',
        'Amount': 'monetary'
    })
    
    print(f"Created RFM features for {len(rfm)} unique customers")
    return rfm

def create_proxy_default(rfm_df, n_segments=3):
    """
    Create proxy default variable using RFM segmentation
    Low RFM score = High risk (default=1)
    High RFM score = Low risk (default=0)
    
    Parameters:
    rfm_df (pd.DataFrame): RFM features dataframe
    n_segments (int): Number of segments for RFM scoring
    
    Returns:
    pd.DataFrame: RFM dataframe with default column added
    """
    rfm_df = rfm_df.copy()
    
    # Create individual scores for R, F, M
    rfm_df['r_score'] = pd.qcut(
        rfm_df['recency'].rank(method='first'), 
        n_segments, 
        labels=False, 
        duplicates='drop'
    )
    rfm_df['f_score'] = pd.qcut(
        rfm_df['frequency'].rank(method='first'), 
        n_segments, 
        labels=False, 
        duplicates='drop'
    )
    rfm_df['m_score'] = pd.qcut(
        rfm_df['monetary'].rank(method='first'), 
        n_segments, 
        labels=False, 
        duplicates='drop'
    )
    
    # Combined RFM score (higher is better customer)
    rfm_df['rfm_score'] = rfm_df['r_score'] + rfm_df['f_score'] + rfm_df['m_score']
    
    # Define risk thresholds
    threshold_bad = rfm_df['rfm_score'].quantile(0.3)
    threshold_good = rfm_df['rfm_score'].quantile(0.8)
    
    # Create default column
    rfm_df['default'] = 1  # Default to high risk
    rfm_df.loc[rfm_df['rfm_score'] >= threshold_good, 'default'] = 0
    
    # Print distribution
    default_counts = rfm_df['default'].value_counts()
    print(f"\nProxy Default Distribution:")
    print(f"Good (0): {default_counts.get(0, 0)} customers ({default_counts.get(0, 0)/len(rfm_df)*100:.1f}%)")
    print(f"Bad (1): {default_counts.get(1, 0)} customers ({default_counts.get(1, 0)/len(rfm_df)*100:.1f}%)")
    
    return rfm_df

