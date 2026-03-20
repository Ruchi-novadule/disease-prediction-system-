import pandas as pd

# 📥 Load dataset
def load_data(path):
    df = pd.read_csv(path)
    return df

# 🧹 Preprocess data
def preprocess_data(df):
    df.fillna(0, inplace=True)
    return df


# ▶️ Run this file
if __name__ == "__main__":
    df = load_data("../data/raw/dataset.csv")
    df = preprocess_data(df)

    print("✅ Dataset Loaded Successfully")
    print("Shape:", df.shape)

    print("\n📊 Columns:")
    print(df.columns)

    print("\n🔍 First 5 rows:")
    print(df.head())