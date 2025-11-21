# Laptop Dashboard README

## 📊 Các Dashboard đã tạo

Dựa trên bài tập Week 4 về Dash interactivity, tôi đã tạo 3 files chính cho dữ liệu laptop ASUS:

### 1. **laptop_interactivity.py** 
Tương tự `dash_interactivity.py` - Dashboard đơn giản

**Tính năng:**
- Dropdown để chọn CPU Category (Intel i9, i7, i5, i3, AMD Ryzen 9, 7, 5, hoặc All)
- Line chart hiển thị Average Price theo RAM
- Tự động cập nhật khi thay đổi filter

**Cách chạy:**
```bash
python laptop_interactivity.py
```
Mở trình duyệt: http://localhost:8050

---

### 2. **laptop_analysis_dashboard.py**
Tương tự `flight_delay.py` - Dashboard phức tạp với nhiều charts

**Tính năng:**
- Dropdown để chọn Price Segment (Budget, Mid-range, High-end, Premium, hoặc All)
- 5 biểu đồ tương tác:
  1. **CPU Distribution**: Bar chart số lượng laptop theo CPU
  2. **GPU Distribution**: Bar chart số lượng laptop theo GPU
  3. **RAM vs Price**: Line chart giá trung bình theo RAM
  4. **Storage vs Price**: Line chart giá trung bình theo Storage
  5. **Price Distribution**: Histogram phân bố giá

**Cách chạy:**
```bash
python laptop_analysis_dashboard.py
```
Mở trình duyệt: http://localhost:8051

---

### 3. **laptop_data_preprocessing.py**
File helper chứa các hàm tiền xử lý dữ liệu

**Tính năng:**
- `load_and_preprocess_laptop_data()`: Đọc và xử lý CSV
- `clean_storage()`: Chuyển đổi storage sang GB
- `classify_cpu()`: Phân loại CPU
- `classify_gpu()`: Phân loại GPU
- `price_segment()`: Tạo phân khúc giá
- `get_laptop_statistics()`: Lấy thống kê tổng quan

**Cách chạy (test):**
```bash
python laptop_data_preprocessing.py
```

---

## 🔧 Cài đặt Dependencies

Trước khi chạy, cần cài đặt các thư viện:

```bash
pip install pandas numpy plotly dash
```

---

## 📁 Cấu trúc Files

```
lab/
├── laptops_asus_data_cellphones_full_v2.csv    # Dữ liệu gốc
├── laptop_interactivity.py                      # Dashboard đơn giản
├── laptop_analysis_dashboard.py                 # Dashboard phức tạp
├── laptop_data_preprocessing.py                 # Helper functions
└── README_laptop_dashboard.md                   # File này
```

---

## 🎯 So sánh với Week 4 Airline Data

| Airline Dashboard | Laptop Dashboard |
|-------------------|------------------|
| Filter: Year | Filter: CPU Category / Price Segment |
| X-axis: Month | X-axis: RAM / Storage |
| Color: Airline | Color: CPU / GPU Category |
| Y-axis: Delay times | Y-axis: Price / Count |
| airline_data.csv | laptops_asus_data_cellphones_full_v2.csv |

---

## ✨ Các cột dữ liệu được tạo

- **Price_VND**: Giá đã làm sạch (numeric)
- **RAM_GB**: RAM đã chuẩn hóa (integer)
- **Storage_GB**: Storage đã chuẩn hóa (float)
- **CPU_Category**: Intel i9/i7/i5/i3, AMD Ryzen 9/7/5, Other
- **GPU_Category**: RTX 4090/4080/4070/4060/4050, RTX 3050, GTX, AMD Radeon, Intel Integrated, Other
- **Price_Segment**: Budget (<15M), Mid-range (15-25M), High-end (25-40M), Premium (>40M)

---

## 🚀 Quick Start

1. **Chạy dashboard đơn giản:**
```bash
cd lab
python laptop_interactivity.py
```

2. **Chạy dashboard phức tạp (khuyến nghị):**
```bash
cd lab
python laptop_analysis_dashboard.py
```

3. **Test preprocessing:**
```bash
python laptop_data_preprocessing.py
```

---

## 💡 Tips

- Hai dashboard chạy trên port khác nhau (8050 và 8051) nên có thể chạy đồng thời
- Dùng Ctrl+C để dừng dashboard
- Refresh browser nếu thay đổi code (hoặc dùng debug=True)
- Tất cả preprocessing được tích hợp sẵn trong mỗi dashboard file

---

## 📝 Bài tập tương ứng

✅ Week 4 - Dash Interactivity  
✅ Plotly Graph Objects & Express  
✅ Callbacks và Dynamic Updates  
✅ Multiple Output Dashboard

---

**Học sinh:** 523H0164  
**Dataset:** ASUS Laptops from Cellphones (790 laptops)  
**Ngày tạo:** November 20, 2025
