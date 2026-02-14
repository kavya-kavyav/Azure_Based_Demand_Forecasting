
print("🔥 STARTING CLEANING...")
import pandas as pd
df = pd.read_csv('azure_large_demand_forecasting_dataset.csv')
print(f"📊 Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("📋 Columns:", df.columns.tolist())

# Quick clean & save
df.drop_duplicates(inplace=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.to_csv('cleaned_data.csv', index=False)
print("✅ SAVED cleaned_data.csv")
print("First 3 rows:")
print(df.head(3))
