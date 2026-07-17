"""
Unit tests for data processing module (Task 4: K-Means Proxy Target)
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath("."))

from src.data_processing import (
    load_raw_data,
    RFMFeatureCreator,
    ProxyTargetCreator,
    create_target_with_clustering,
    test_kmeans_target_creation
)


def test_load_raw_data():
    """Test data loading"""
    try:
        df = load_raw_data("data/raw/germany.csv")
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
    except FileNotFoundError:
        pytest.skip("Data file not found")


def test_rfm_feature_creator():
    """Test RFM feature creation"""
    sample_data = pd.DataFrame({
        'CustomerId': ['A', 'A', 'B', 'B'],
        'TransactionId': [1, 2, 3, 4],
        'Amount': [100, 200, 50, 150],
        'TransactionStartTime': ['2024-01-01', '2024-01-15', '2024-01-10', '2024-01-20']
    })
    
    transformer = RFMFeatureCreator()
    result = transformer.transform(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert 'recency' in result.columns
    assert 'frequency' in result.columns
    assert 'monetary' in result.columns
    assert len(result) == 2


def test_proxy_target_creator():
    """Test K-Means proxy target creation (TASK 4)"""
    sample_rfm = pd.DataFrame({
        'recency': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'frequency': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
        'monetary': [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
    })
    
    creator = ProxyTargetCreator(n_clusters=3, random_state=42)
    result = creator.fit_transform(sample_rfm)
    
    assert isinstance(result, pd.DataFrame)
    assert 'is_high_risk' in result.columns
    assert result['is_high_risk'].isin([0, 1]).all()
    assert len(result) == 10


def test_kmeans_target_creation_integration():
    """Test complete K-Means target creation pipeline (TASK 4)"""
    sample_data = pd.DataFrame({
        'CustomerId': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'],
        'TransactionId': range(1, 11),
        'Amount': [100, 200, 50, 150, 300, 400, 10, 20, 500, 600],
        'TransactionStartTime': [
            '2024-01-01', '2024-01-15',
            '2024-01-10', '2024-01-20',
            '2024-02-01', '2024-02-15',
            '2024-03-01', '2024-03-15',
            '2024-01-05', '2024-01-25'
        ],
        'FraudResult': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'ChannelId': ['web', 'web', 'mobile', 'mobile', 'web', 'app', 'web', 'web', 'mobile', 'web'],
        'ProductCategory': ['electronics', 'electronics', 'clothing', 'clothing', 'books', 'books',
                           'electronics', 'electronics', 'clothing', 'clothing']
    })
    
    target = create_target_with_clustering(sample_data, n_clusters=3, random_state=42)
    
    assert isinstance(target, pd.Series)
    assert target.name == 'is_high_risk'
    assert len(target) == 5  # 5 unique customers
    assert target.isin([0, 1]).all()


def test_reproducibility():
    """Test that K-Means target creation is reproducible"""
    sample_rfm = pd.DataFrame({
        'recency': np.random.randint(1, 100, 50),
        'frequency': np.random.randint(1, 100, 50),
        'monetary': np.random.randint(100, 1000, 50)
    })
    
    creator1 = ProxyTargetCreator(n_clusters=3, random_state=42)
    result1 = creator1.fit_transform(sample_rfm)
    
    creator2 = ProxyTargetCreator(n_clusters=3, random_state=42)
    result2 = creator2.fit_transform(sample_rfm)
    
    pd.testing.assert_series_equal(result1['is_high_risk'], result2['is_high_risk'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
