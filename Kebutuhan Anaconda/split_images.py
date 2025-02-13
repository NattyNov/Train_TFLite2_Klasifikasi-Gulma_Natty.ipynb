import os
import shutil
from sklearn.model_selection import train_test_split

# Path ke folder gambar
image_folder = 'images'  # Gantilah dengan lokasi folder gambar Anda
output_folder = 'output'  # Folder output untuk menyimpan train, test, dan validation

# Membuat folder untuk train, test, dan validation jika belum ada
os.makedirs(os.path.join(output_folder, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'test'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'validation'), exist_ok=True)

# Mendapatkan semua file gambar (.jpg, .png, .jpeg)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Pisahkan 80% data untuk training dan sisanya (20%) untuk test + validasi
train_files, temp_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Pisahkan 50% dari data yang tersisa untuk test dan 50% untuk validasi (20% * 0.5 = 10%)
test_files, val_files = train_test_split(temp_files, test_size=0.5, random_state=42)  # 0.5 * 0.2 = 0.1

# Fungsi untuk memindahkan file gambar ke folder yang sesuai
def move_files(file_list, folder):
    for file in file_list:
        image_path = os.path.join(image_folder, file)
        shutil.move(image_path, os.path.join(output_folder, folder, file))

# Memindahkan file gambar ke folder yang sesuai
move_files(train_files, 'train')
move_files(test_files, 'test')
move_files(val_files, 'validation')

# Tampilkan hasil pembagian
print("Dataset berhasil dibagi:")
print(f"Train: {len(train_files)} gambar")
print(f"Test: {len(test_files)} gambar")
print(f"Validation: {len(val_files)} gambar")
