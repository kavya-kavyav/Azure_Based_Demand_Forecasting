import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("🎯 MILESTONE 2: FEATURE ENGINEERING")
print("Transforming cleaned_data.csv → model_ready_features.csv")

# 1. Load cleaned data
df = pd.read_csv('cleaned_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
print(f"✅ Loaded: {len(df)} rows")

# 2. TIME-BASED FEATURES (Seasonality & Trends)
print("\n⏰ Creating time features...")
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # 1 = weekend
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

# Seasonality flags (Q1-Q4 business cycles)
df['season_q1'] = (df['quarter'] == 1).astype(int)
df['season_q2'] = (df['quarter'] == 2).astype(int)
df['season_q3'] = (df['quarter'] == 3).astype(int)
df['season_q4'] = (df['quarter'] == 4).astype(int)

# 3. LAG FEATURES (Historical demand patterns)
print("📈 Creating lag features...")
lags = [1, 7, 30]  # 1-day, weekly, monthly lags
for lag in lags:
    df[f'demand_lag_{lag}'] = df['demand_units'].shift(lag)
    df[f'demand_mean_lag_{lag}'] = df['demand_units'].rolling(window=lag).mean().shift(1)

# Rolling statistics (trends)
df['demand_rolling_mean_7'] = df['demand_units'].rolling(7).mean()
df['demand_rolling_std_7'] = df['demand_units'].rolling(7).std()

# 4. USAGE TRENDS & SPIKES (Azure metrics)
print("☁️ Engineering Azure usage features...")
if 'azure_cpu_utilization_percent' in df.columns:
    df['cpu_spike'] = (df['azure_cpu_utilization_percent'] > df['azure_cpu_utilization_percent'].quantile(0.9)).astype(int)
    df['cpu_trend'] = df['azure_cpu_utilization_percent'].rolling(3).mean()

if 'azure_memory_utilization_percent' in df.columns:
    df['memory_spike'] = (df['azure_memory_utilization_percent'] > df['azure_memory_utilization_percent'].quantile(0.9)).astype(int)
    df['memory_trend'] = df['azure_memory_utilization_percent'].rolling(3).mean()

# Active instances trend
if 'active_instances' in df.columns:
    df['instances_trend'] = df['active_instances'].rolling(7).mean()
    df['high_utilization'] = ((df['azure_cpu_utilization_percent'] > 70) & 
                             (df['azure_memory_utilization_percent'] > 70)).astype(int)

# 5. USER BEHAVIOR & BUSINESS FEATURES
print("👥 Creating user behavior features...")
if 'marketing_spend_usd' in df.columns:
    df['marketing_roi'] = df['demand_units'] / (df['marketing_spend_usd'] + 1)  # Avoid div0
    df['high_marketing'] = (df['marketing_spend_usd'] > df['marketing_spend_usd'].quantile(0.75)).astype(int)

if 'competitor_index' in df.columns:
    df['competitor_threat'] = (df['competitor_index'] > 0.8).astype(int)

# Demand growth rate
df['demand_growth'] = df['demand_units'].pct_change().fillna(0)

# 6. SERVICE UPTIME & RELIABILITY
df['azure_load'] = df.get('azure_cpu_utilization_percent', 0) + df.get('azure_memory_utilization_percent', 0)
df['service_stress'] = (df['azure_load'] > df['azure_load'].quantile(0.85)).astype(int)

# 7. RESHAPE TO MODEL-READY FORMAT
print("🔄 Reshaping for modeling...")

# Select model features (drop raw categoricals, keep encoded)
model_features = ['demand_units']  # Target first
numeric_cols = df.select_dtypes(include=['number']).columns
model_features.extend([col for col in numeric_cols if col not in ['index', 'demand_units']])

df_model = df[model_features].copy()
df_model = df_model.dropna()  # Remove rows with NaN from lags

# 8. FEATURE SCALING (Standardization)
scaler = StandardScaler()
feature_cols = [col for col in df_model.columns if col != 'demand_units']
df_model[feature_cols] = scaler.fit_transform(df_model[feature_cols])

print(f"✅ Model-ready shape: {df_model.shape}")
print("\n📋 Model Features Created:")
for i, col in enumerate(df_model.columns, 1):
    print(f"  {i:2d}. {col}")

# SAVE RESULTS
df_model.to_csv('model_ready_features.csv', index=False)
print(f"\n💾 SAVED: model_ready_features.csv ({len(df_model)} rows × {len(df_model.columns)} features)")

# Feature importance preview (correlation with target)
print("\n🔥 TOP 10 DEMAND-DRIVING FEATURES:")
correlations = df_model.corr()['demand_units'].abs().sort_values(ascending=False)[1:11]
print(correlations.round(3))

