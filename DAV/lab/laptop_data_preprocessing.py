"""
Laptop Data Preprocessing Helper
Tệp này chứa các hàm tiền xử lý dữ liệu laptop để sử dụng trong các dashboard Dash
"""

import pandas as pd
import numpy as np
import re


def load_and_preprocess_laptop_data(csv_path='laptops_asus_data_cellphones_full_v2.csv'):
    """
    Đọc và tiền xử lý dữ liệu laptop từ file CSV
    
    Args:
        csv_path (str): Đường dẫn đến file CSV
        
    Returns:
        pandas.DataFrame: DataFrame đã được xử lý với các cột mới
    """
    # Đọc dữ liệu
    df = pd.read_csv(csv_path)
    
    # 1. Xử lý cột Price
    df['Price_VND'] = (
        df['Price']
        .astype(str)
        .str.replace('đ', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.strip()
    )
    df['Price_VND'] = pd.to_numeric(df['Price_VND'], errors='coerce')
    
    # 2. Xử lý cột RAM
    df['RAM_GB'] = (
        df['RAM']
        .astype(str)
        .str.replace('GB', '', regex=False)
        .str.strip()
    )
    df['RAM_GB'] = pd.to_numeric(df['RAM_GB'], errors='coerce').astype('Int64')
    
    # 3. Xử lý cột Storage
    df['Storage_GB'] = df['Storage'].astype(str).apply(clean_storage)
    
    # 4. Phân loại CPU
    df['CPU_Category'] = df['CPU'].apply(classify_cpu)
    
    # 5. Phân loại GPU
    df['GPU_Category'] = df['GPU'].apply(classify_gpu)
    
    # 6. Tạo phân khúc giá
    df['Price_Segment'] = df['Price_VND'].apply(price_segment)
    
    # 7. Loại bỏ các dòng thiếu dữ liệu quan trọng
    df = df.dropna(subset=['Price_VND', 'RAM_GB'])
    
    print(f"✅ Đã xử lý xong {len(df)} laptops")
    print(f"📊 Các cột mới được tạo: Price_VND, RAM_GB, Storage_GB, CPU_Category, GPU_Category, Price_Segment")
    
    return df


def clean_storage(storage_str):
    """
    Chuyển đổi chuỗi storage thành số GB
    
    Args:
        storage_str (str): Chuỗi storage (vd: "512GB", "1TB")
        
    Returns:
        float: Dung lượng storage tính bằng GB
    """
    if pd.isna(storage_str) or storage_str.upper() == 'NAN':
        return np.nan
    
    storage_str = storage_str.upper()
    
    # Kiểm tra TB
    tb_match = re.search(r'(\d+)\s*TB', storage_str)
    if tb_match:
        return float(tb_match.group(1)) * 1024
    
    # Kiểm tra GB
    gb_match = re.search(r'(\d+)\s*GB', storage_str)
    if gb_match:
        return float(gb_match.group(1))
    
    return np.nan


def classify_cpu(cpu_str):
    """
    Phân loại CPU thành các category chuẩn
    
    Args:
        cpu_str (str): Chuỗi mô tả CPU
        
    Returns:
        str: Category của CPU
    """
    if pd.isna(cpu_str):
        return 'Unknown'
    
    cpu_str = str(cpu_str).lower()
    
    # Intel processors
    if 'core i9' in cpu_str or 'core 9' in cpu_str:
        return 'Intel Core i9'
    elif 'core i7' in cpu_str or 'core 7' in cpu_str:
        return 'Intel Core i7'
    elif 'core i5' in cpu_str or 'core 5' in cpu_str:
        return 'Intel Core i5'
    elif 'core i3' in cpu_str or 'core 3' in cpu_str:
        return 'Intel Core i3'
    
    # AMD processors
    elif 'ryzen 9' in cpu_str:
        return 'AMD Ryzen 9'
    elif 'ryzen 7' in cpu_str:
        return 'AMD Ryzen 7'
    elif 'ryzen 5' in cpu_str:
        return 'AMD Ryzen 5'
    elif 'ryzen 3' in cpu_str:
        return 'AMD Ryzen 3'
    
    else:
        return 'Other'


def classify_gpu(gpu_str):
    """
    Phân loại GPU thành các category chuẩn
    
    Args:
        gpu_str (str): Chuỗi mô tả GPU
        
    Returns:
        str: Category của GPU
    """
    if pd.isna(gpu_str):
        return 'Unknown'
    
    gpu_str = str(gpu_str).lower()
    
    # NVIDIA RTX 4000 series
    if 'rtx 4090' in gpu_str:
        return 'RTX 4090'
    elif 'rtx 4080' in gpu_str:
        return 'RTX 4080'
    elif 'rtx 4070' in gpu_str:
        return 'RTX 4070'
    elif 'rtx 4060' in gpu_str:
        return 'RTX 4060'
    elif 'rtx 4050' in gpu_str:
        return 'RTX 4050'
    
    # NVIDIA RTX 3000 series
    elif 'rtx 3050' in gpu_str:
        return 'RTX 3050'
    elif 'rtx' in gpu_str:
        return 'RTX Other'
    
    # NVIDIA GTX series
    elif 'gtx' in gpu_str:
        return 'GTX'
    
    # AMD GPUs
    elif 'radeon' in gpu_str:
        return 'AMD Radeon'
    
    # Integrated GPUs
    elif 'intel' in gpu_str or 'iris' in gpu_str or 'uhd' in gpu_str:
        return 'Intel Integrated'
    
    else:
        return 'Other'


def price_segment(price):
    """
    Phân loại laptop theo phân khúc giá
    
    Args:
        price (float): Giá laptop (VND)
        
    Returns:
        str: Phân khúc giá
    """
    if pd.isna(price):
        return 'Unknown'
    elif price < 15000000:
        return 'Budget (<15M)'
    elif price < 25000000:
        return 'Mid-range (15-25M)'
    elif price < 40000000:
        return 'High-end (25-40M)'
    else:
        return 'Premium (>40M)'


def get_laptop_statistics(df):
    """
    Lấy thống kê tổng quan về dữ liệu laptop
    
    Args:
        df (pandas.DataFrame): DataFrame đã được xử lý
        
    Returns:
        dict: Dictionary chứa các thống kê
    """
    stats = {
        'total_laptops': len(df),
        'avg_price': df['Price_VND'].mean(),
        'min_price': df['Price_VND'].min(),
        'max_price': df['Price_VND'].max(),
        'cpu_categories': df['CPU_Category'].nunique(),
        'gpu_categories': df['GPU_Category'].nunique(),
        'ram_options': sorted(df['RAM_GB'].dropna().unique().tolist()),
        'price_segments': df['Price_Segment'].value_counts().to_dict()
    }
    
    return stats


# Example usage
if __name__ == '__main__':
    # Test the preprocessing
    df = load_and_preprocess_laptop_data()
    
    print("\n📈 Thống kê dữ liệu:")
    stats = get_laptop_statistics(df)
    
    print(f"   Tổng số laptop: {stats['total_laptops']}")
    print(f"   Giá trung bình: {stats['avg_price']:,.0f} VND")
    print(f"   Giá thấp nhất: {stats['min_price']:,.0f} VND")
    print(f"   Giá cao nhất: {stats['max_price']:,.0f} VND")
    print(f"   Số loại CPU: {stats['cpu_categories']}")
    print(f"   Số loại GPU: {stats['gpu_categories']}")
    print(f"   RAM options: {stats['ram_options']}")
    print(f"\n   Phân bố phân khúc giá:")
    for segment, count in stats['price_segments'].items():
        print(f"      {segment}: {count} laptops")
