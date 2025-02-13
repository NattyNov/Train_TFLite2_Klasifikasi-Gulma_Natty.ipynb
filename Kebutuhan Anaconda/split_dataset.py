import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset dari dua file CSV
train_file = "train_labels.csv"  # Gantilah dengan nama file train_labels.csv Anda
test_file = "test_labels.csv"  # Gantilah dengan nama file test_labels.csv Anda

# Membaca kedua file
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Gabungkan data train dan test menjadi satu dataframe
df = pd.concat([train_df, test_df], ignore_index=True)

# Pisahkan 80% data untuk training dan sisanya (20%) untuk test + validasi
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Pisahkan 50% dari data yang tersisa untuk test dan 50% untuk validasi (20% * 0.5 = 10%)
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Simpan dataset yang telah dibagi
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
val_df.to_csv("val.csv", index=False)

# Tampilkan hasil pembagian
print("Dataset berhasil dibagi:")
print(f"Train: {len(train_df)} data")
print(f"Test: {len(test_df)} data")
print(f"Validation: {len(val_df)} data")
