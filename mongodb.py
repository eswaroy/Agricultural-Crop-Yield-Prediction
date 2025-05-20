from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["agriculture_db"]
collection = db["crop_data"]

# Load cleaned dataset
df = pd.read_csv(r"C:\tmp\final_crop_data.csv")

# Drop rows with missing data
df.dropna(inplace=True)

# Convert DataFrame to dictionary and insert
records = df.to_dict(orient="records")
collection.delete_many({})  # Clean old data
collection.insert_many(records)

print(f"✅ Inserted {len(records)} records into MongoDB.")
