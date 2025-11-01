import kagglehub
import os
import shutil

# Create datasets directory if it doesn't exist
os.makedirs("datasets", exist_ok=True)

# Download latest version of insurance dataset
print("Downloading insurance dataset...")
path = kagglehub.dataset_download("mirichoi0218/insurance")
print("Downloaded to:", path)

# Copy files to datasets folder
for file in os.listdir(path):
    src = os.path.join(path, file)
    dst = os.path.join("datasets", file)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"Copied {file} to datasets/")

# Download latest version of car price prediction dataset
print("\nDownloading car price prediction dataset...")
path = kagglehub.dataset_download("hellbuoy/car-price-prediction")
print("Downloaded to:", path)





# Copy files to datasets folder
for file in os.listdir(path):
    src = os.path.join(path, file)
    dst = os.path.join("datasets", file)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"Copied {file} to datasets/")

# Download latest version of used handheld device dataset
print("\nDownloading used handheld device dataset...")
path = kagglehub.dataset_download("ahsan81/used-handheld-device-data")
print("Downloaded to:", path)

# Copy files to datasets folder
for file in os.listdir(path):
    src = os.path.join(path, file)
    dst = os.path.join("datasets", file)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"Copied {file} to datasets/")
