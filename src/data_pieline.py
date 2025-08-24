# Load your combined dataset# Cell 1: Import Libraries and Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load your combined dataset
df = pd.read_csv('trainingdataset.csv')  # Replace with your file path

print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nFirst few rows:")
df.head()

import pandas as pd

# Load the dataset
df = pd.read_csv('trainingdataset.csv')

# --- Data Preprocessing based on your CSV columns ---

# Let's identify key features from your dataset for A and B.
# We will create congestion labels and features for both '_A' and '_B' entities.

# Target for classification (congestion)
congestion_threshold_bandwidth = 0.8  # 80% bandwidth usage
congestion_threshold_latency = 50     # Adjust based on your data

# Create congestion label for A
# NOTE: The CSV does not have a 'Bandwidth (MB/s)_A'. We will assume 'Bandwidth Used (MB/s)_A' is the utilization metric.
# To create a utilization percentage, we would need the *total* bandwidth.
# Since we don't have total bandwidth, we'll define congestion based on latency and high bandwidth *usage*.
# We'll consider high usage to be a value above the 80th percentile for this exercise.

high_bandwidth_usage_threshold_A = df['Bandwidth Used (MB/s)_A'].quantile(0.80)
df['is_congested_A'] = ((df['Bandwidth Used (MB/s)_A'] > high_bandwidth_usage_threshold_A) |
                       (df['Latency (ms)_A'] > congestion_threshold_latency)).astype(int)

# Create congestion label for B
high_bandwidth_usage_threshold_B = df['Bandwidth Used (MB/s)_B'].quantile(0.80)
df['is_congested_B'] = ((df['Bandwidth Used (MB/s)_B'] > high_bandwidth_usage_threshold_B) |
                       (df['Latency (ms)_B'] > congestion_threshold_latency)).astype(int)


# Handle date-time features.
# The dataset already has 'Hour' and 'Date' and 'Month'
# We will create 'day_of_week' and 'is_weekend' from the 'Date' and 'Month'.
# We need to construct a full date string to convert to datetime. Let's assume the year is 2024 for this purpose.
df['year'] = 2024 # Assuming a recent year since not provided
df['Date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Date'].astype(str))

df['day_of_week'] = df['Date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Define feature columns based on your CSV
feature_columns = [
    'Traffic Volume (MB/s)_A', 'Latency (ms)_A', 'Bandwidth Used (MB/s)_A',
    'Traffic Volume (MB/s)_B', 'Latency (ms)_B', 'Bandwidth Used (MB/s)_B',
    'Hour', 'day_of_week', 'is_weekend',
    'Traffic Volume (MB/s)_A_mean', 'Traffic Volume (MB/s)_A_std',
    'Latency (ms)_A_mean', 'Latency (ms)_A_std',
    'Bandwidth Used (MB/s)_A_mean', 'Bandwidth Used (MB/s)_A_std',
    'Traffic Volume (MB/s)_B_mean', 'Traffic Volume (MB/s)_B_std',
    'Latency (ms)_B_mean', 'Latency (ms)_B_std',
    'Bandwidth Used (MB/s)_B_mean', 'Bandwidth Used (MB/s)_B_std'
]

# Select features for modeling
features = df[feature_columns].copy()

# For this example, we will use 'is_congested_A' as the target.
targets_classification = df['is_congested_A'].values

# The regression target could be future traffic volume.
# For this example, we will predict the next time step's traffic volume for A.
targets_regression = df['Traffic Volume (MB/s)_A'].shift(-1).fillna(method='ffill').values

print("Congestion Distribution (A):")
print(df['is_congested_A'].value_counts(normalize=True))

print("\nCongestion Distribution (B):")
print(df['is_congested_B'].value_counts(normalize=True))

print(f"\nFeatures shape: {features.shape}")
print(f"Features: {feature_columns}")

# Handle missing values using forward and backward fill
features = features.fillna(method='ffill').fillna(method='bfill')

# Display basic statistics of the features
print("\nFeature Statistics:")
print(features.describe())