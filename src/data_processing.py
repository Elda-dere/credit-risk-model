"""
Task 4: Complete Feature Engineering Pipeline with K-Means Proxy Target

This module implements:
1. Feature engineering pipeline (RFM, Aggregate, Time, Fraud, One-Hot)
2. K-Means clustering for proxy target creation
3. Full sklearn Pipeline for reproducibility

Task 4 Focus: K-Means clustering to identify high-risk customers
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CUSTOM TRANSFORMERS FOR FEATURE ENGINEERING
# ============================================================

class RFMFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates RFM (Recency, Frequency, Monetary) features
    - Recency: Days since last transaction
    - Frequency: Number of transactions
    - Monetary: Total amount spent (positive transactions only)
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        max_date = X['TransactionStartTime'].max()
        
        rfm = X.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (max_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': lambda x: x[x > 0].sum()
        }).rename(columns={
            'TransactionStartTime': 'recency',
            'TransactionId': 'frequency',
            'Amount': 'monetary'
        })
        return rfm


class ProxyTargetCreator(BaseEstimator, TransformerMixin):
    """
    Creates proxy default target using K-Means clustering (TASK 4)
    
    Steps:
    1. Scale RFM features using StandardScaler
    2. Apply K-Means clustering (n=3, random_state=42)
    3. Identify high-risk cluster (low frequency, low monetary, high recency)
    4. Assign is_high_risk = 1 to high-risk cluster, 0 to others
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.high_risk_cluster = None
    
    def fit(self, X, y=None):
        # X should be RFM features (recency, frequency, monetary)
        rfm_scaled = self.scaler.fit_transform(X[['recency', 'frequency', 'monetary']])
        self.kmeans.fit(rfm_scaled)
        
        # Identify high-risk cluster
        cluster_centers = self.kmeans.cluster_centers_
        cluster_scores = []
        for i in range(self.n_clusters):
            # High risk = low frequency, low monetary, high recency
            score = -cluster_centers[i][1] - cluster_centers[i][2] + cluster_centers[i][0]
            cluster_scores.append(score)
        self.high_risk_cluster = np.argmax(cluster_scores)
        print(f"High-risk cluster identified: Cluster {self.high_risk_cluster}")
        return self
    
    def transform(self, X):
        rfm_scaled = self.scaler.transform(X[['recency', 'frequency', 'monetary']])
        clusters = self.kmeans.predict(rfm_scaled)
        
        # Create target: 1 = high risk, 0 = low risk
        target = (clusters == self.high_risk_cluster).astype(int)
        
        print(f"Target distribution:\n{pd.Series(target).value_counts(normalize=True)}")
        return pd.DataFrame({'is_high_risk': target}, index=X.index)


class AggregateFeatureCreator(BaseEstimator, TransformerMixin):
    """Creates aggregate features: total_amount, avg_amount, transaction_count, std_amount"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['Amount'] = pd.to_numeric(X['Amount'], errors='coerce')
        
        agg = X.groupby('CustomerId').agg({
            'Amount': [
                ('total_amount', lambda x: x[x > 0].sum()),
                ('avg_amount', lambda x: x[x > 0].mean()),
                ('count_transactions', 'count'),
                ('std_amount', lambda x: x[x > 0].std())
            ]
        })
        agg.columns = ['total_amount', 'avg_amount', 'transaction_count', 'std_amount']
        agg['std_amount'] = agg['std_amount'].fillna(0)
        return agg


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts time-based features: hour, day, month, year, dayofweek"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        first = X.groupby('CustomerId')['TransactionStartTime'].first()
        
        time_features = pd.DataFrame(index=first.index)
        time_features['transaction_hour'] = first.dt.hour
        time_features['transaction_day'] = first.dt.day
        time_features['transaction_month'] = first.dt.month
        time_features['transaction_year'] = first.dt.year
        time_features['transaction_dayofweek'] = first.dt.dayofweek
        return time_features


class FraudFeatureAggregator(BaseEstimator, TransformerMixin):
    """Aggregates fraud features: total_fraud, fraud_rate"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        fraud = X.groupby('CustomerId').agg({
            'FraudResult': [('total_fraud', 'sum'), ('fraud_rate', 'mean')]
        })
        fraud.columns = ['total_fraud', 'fraud_rate']
        fraud['fraud_rate'] = fraud['fraud_rate'].fillna(0)
        return fraud


class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    """One-hot encodes categorical variables: ChannelId, ProductCategory"""
    
    def __init__(self):
        self.categorical_cols = ['ChannelId', 'ProductCategory']
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        X = X.copy()
        first = X.groupby('CustomerId')[self.categorical_cols].first()
        encoded = pd.get_dummies(first, columns=self.categorical_cols)
        self.feature_names_ = encoded.columns.tolist()
        return self
    
    def transform(self, X):
        X = X.copy()
        first = X.groupby('CustomerId')[self.categorical_cols].first()
        encoded = pd.get_dummies(first, columns=self.categorical_cols)
        for col in self.feature_names_:
            if col not in encoded.columns:
                encoded[col] = 0
        return encoded[self.feature_names_]


class FeatureMerger(BaseEstimator, TransformerMixin):
    """Merges all feature sets into a single DataFrame"""
    
    def __init__(self):
        self.feature_generators = [
            ('rfm', RFMFeatureCreator()),
            ('aggregate', AggregateFeatureCreator()),
            ('time', TimeFeatureExtractor()),
            ('fraud', FraudFeatureAggregator()),
            ('categorical', OneHotEncoderTransformer())
        ]
    
    def fit(self, X, y=None):
        for name, transformer in self.feature_generators:
            transformer.fit(X, y)
        return self
    
    def transform(self, X):
        result = pd.DataFrame(index=X['CustomerId'].unique())
        for name, transformer in self.feature_generators:
            features = transformer.transform(X)
            result = result.join(features)
        return result


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handles missing values using median imputation"""
    
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.imputer_ = SimpleImputer(strategy='median')
        self.imputer_.fit(X[numeric_cols])
        self.numeric_cols_ = numeric_cols
        return self
    
    def transform(self, X):
        imputed = self.imputer_.transform(X[self.numeric_cols_])
        return pd.DataFrame(imputed, columns=self.numeric_cols_, index=X.index)


class NumericalScaler(BaseEstimator, TransformerMixin):
    """Standardizes numerical features (mean=0, std=1)"""
    
    def fit(self, X, y=None):
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)
        self.feature_names_ = X.columns.tolist()
        return self
    
    def transform(self, X):
        scaled = self.scaler_.transform(X)
        return pd.DataFrame(scaled, columns=self.feature_names_, index=X.index)


# ============================================================
# MAIN FUNCTIONS
# ============================================================

def load_raw_data(file_path):
    """Load raw transaction data"""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df


def create_target_with_clustering(df, n_clusters=3, random_state=42):
    """
    TASK 4: Create proxy default target using K-Means clustering
    
    Parameters:
    df: Raw transaction data
    n_clusters: Number of clusters (default=3)
    random_state: For reproducibility (default=42)
    
    Returns:
    target: Series with is_high_risk labels (1=high risk, 0=low risk)
    """
    print("="*60)
    print("TASK 4: Creating Proxy Target with K-Means Clustering")
    print("="*60)
    
    # Step 1: Create RFM features
    print("\n1. Creating RFM features...")
    rfm_creator = RFMFeatureCreator()
    rfm = rfm_creator.transform(df)
    print(f"   RFM shape: {rfm.shape}")
    print(f"   RFM columns: {rfm.columns.tolist()}")
    
    # Step 2: Scale RFM features
    print("\n2. Scaling RFM features...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
    print(f"   Scaled RFM shape: {rfm_scaled.shape}")
    
    # Step 3: Apply K-Means clustering
    print(f"\n3. Applying K-Means clustering (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    print(f"   Cluster distribution:\n{pd.Series(clusters).value_counts().sort_index()}")
    
    # Step 4: Identify high-risk cluster
    print("\n4. Identifying high-risk cluster...")
    cluster_centers = kmeans.cluster_centers_
    print("   Cluster centers (recency, frequency, monetary):")
    for i, center in enumerate(cluster_centers):
        print(f"   Cluster {i}: recency={center[0]:.2f}, frequency={center[1]:.2f}, monetary={center[2]:.2f}")
    
    cluster_scores = []
    for i in range(n_clusters):
        # High risk = low frequency, low monetary, high recency
        score = -cluster_centers[i][1] - cluster_centers[i][2] + cluster_centers[i][0]
        cluster_scores.append(score)
    high_risk_cluster = np.argmax(cluster_scores)
    print(f"\n   High-risk cluster: Cluster {high_risk_cluster}")
    
    # Step 5: Create target variable
    print("\n5. Creating target variable (is_high_risk)...")
    target = (clusters == high_risk_cluster).astype(int)
    target_series = pd.Series(target, index=rfm.index, name='is_high_risk')
    
    print(f"\n   Target distribution:")
    print(f"   Low risk (0): {len(target_series[target_series == 0])} customers ({target_series[target_series == 0].count()/len(target_series)*100:.1f}%)")
    print(f"   High risk (1): {len(target_series[target_series == 1])} customers ({target_series[target_series == 1].count()/len(target_series)*100:.1f}%)")
    
    return target_series


def create_feature_pipeline():
    """Create the complete feature engineering pipeline"""
    pipeline = Pipeline([
        ('feature_merger', FeatureMerger()),
        ('missing_handler', MissingValueHandler()),
        ('scaler', NumericalScaler())
    ])
    return pipeline


def prepare_model_ready_data(df, n_clusters=3, random_state=42):
    """
    Prepare model-ready data with features and target
    
    Parameters:
    df: Raw transaction data
    n_clusters: Number of clusters for target creation
    random_state: For reproducibility
    
    Returns:
    features: Model-ready features DataFrame
    target: Target Series (is_high_risk)
    """
    print("\n" + "="*60)
    print("PREPARING MODEL-READY DATA")
    print("="*60)
    
    # Step 1: Create target using clustering (TASK 4)
    print("\n[Task 4] Creating target with K-Means...")
    target = create_target_with_clustering(df, n_clusters=n_clusters, random_state=random_state)
    
    # Step 2: Create feature pipeline
    print("\n[Task 3] Creating feature pipeline...")
    pipeline = create_feature_pipeline()
    features = pipeline.fit_transform(df)
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"Features columns: {features.columns.tolist()}")
    print(f"Target shape: {target.shape}")
    print(f"Target distribution:\n{target.value_counts(normalize=True)}")
    
    return features, target


# ============================================================
# TEST FUNCTIONS
# ============================================================

def test_kmeans_target_creation():
    """
    Simple test to verify K-Means target creation works
    """
    print("\n" + "="*60)
    print("TESTING K-MEANS TARGET CREATION")
    print("="*60)
    
    # Create sample data
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
    
    # Create target
    target = create_target_with_clustering(sample_data, n_clusters=3, random_state=42)
    
    print(f"\n✅ Test passed! Target created with {len(target)} customers")
    print(f"Target distribution:\n{target.value_counts(normalize=True)}")
    return target


if __name__ == "__main__":
    print("Testing Task 4: K-Means Proxy Target Creation")
    test_kmeans_target_creation()
