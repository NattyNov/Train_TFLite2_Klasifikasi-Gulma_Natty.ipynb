import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset dari CSV
csv_file = "data.csv"  # Gantilah dengan nama file dataset Anda
df = pd.read_csv(csv_file)

# Pisahkan 80% data untuk training dan sisanya (20%) untuk test + validasi
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Pisahkan 10% untuk test dan 10% untuk validasi dari data yang tersisa
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Simpan dataset yang telah dibagi
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
val_df.to_csv("val.csv", index=False)

print("Dataset berhasil dibagi:")
print(f"Train: {len(train_df)} data")
print(f"Test: {len(test_df)} data")
print(f"Validation: {len(val_df)} data")