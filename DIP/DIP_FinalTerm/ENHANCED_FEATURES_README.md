# Enhanced Traffic Sign Detection System

## 🚀 Tính Năng Mới

### 1. Feature Matching
- **Mục đích**: Nhận diện loại biển báo cụ thể (stop, no entry, speed limit, etc.)
- **Công nghệ**: SIFT/ORB/AKAZE feature detection + Brute Force Matcher
- **Phương pháp**:
  - Trích xuất keypoints và descriptors từ template và ROI
  - Matching sử dụng knnMatch với k=2
  - Áp dụng Lowe's Ratio Test (threshold = 0.75)
  - Tính confidence score dựa trên số matches và distance

### 2. Color Ratio Analysis
```python
calculate_color_ratio(roi, color_params) -> float
```
- **Mục đích**: Xác định tỷ lệ diện tích màu đặc trưng trong biển báo
- **Phương pháp**:
  - Chuyển đổi sang HSV color space
  - Tạo mask với color range
  - Tính tỷ lệ: `color_pixels / total_pixels`
- **Threshold**: 
  - Blue: ≥ 15%
  - Red: ≥ 20%
  - Yellow: ≥ 12%

### 3. Edge Density Analysis
```python
calculate_edge_density(roi) -> float
```
- **Mục đích**: Đo độ phức tạp biên của biển báo
- **Phương pháp**:
  - Canny edge detection
  - Tính tỷ lệ: `edge_pixels / total_pixels`
- **Threshold**: ≥ 0.05 (biển báo thường có nhiều cạnh)

### 4. Aspect Ratio Filtering
```python
check_aspect_ratio(aspect_ratio, expected_range) -> bool
```
- **Mục đích**: Lọc các detection không đúng tỷ lệ hình dạng
- **Expected ranges**:
  - Circle signs: 0.8 - 1.2 (gần vuông)
  - Triangle signs: 0.7 - 1.5 (có thể cao hơn rộng)

### 5. Texture Analysis
```python
calculate_texture_features(roi) -> Dict
```
- **Metrics**:
  - **Contrast**: std_dev / mean (độ tương phản)
  - **Uniformity**: 1/(entropy + 1) (độ đồng đều)
  - **Standard Deviation**: Độ phân tán giá trị pixel
  - **Entropy**: Độ phức tạp thông tin

## 📊 Cấu Trúc Classes

```
TrafficSignConfig
├── Template paths
├── Feature matching settings
└── Enhanced DIP parameters

FeatureMatchingEngine
├── Load templates
├── Extract features
└── Match with ROI

EnhancedDIPAnalyzer
├── Color ratio
├── Edge density
├── Aspect ratio
├── Texture analysis
└── Combined validation

EnhancedTrafficSignDetector
├── Preprocessing
├── Color segmentation
├── Shape detection
├── DIP validation
└── Feature matching

TemporalSignFilter
└── (Enhanced with sign_type tracking)

EnhancedVisualizer
└── Display all metrics
```

## 🎯 Pipeline Xử Lý

```
1. Preprocessing (CLAHE + Saturation boost)
        ↓
2. Color Segmentation (HSV range + Morphology)
        ↓
3. Shape Detection (Contours + Geometry)
        ↓
4. DIP Validation
   ├── Color Ratio ≥ threshold?
   ├── Aspect Ratio in range?
   ├── Edge Density ≥ 0.05?
   └── Texture valid?
        ↓
5. Feature Matching (if enabled)
   ├── Extract keypoints
   ├── Match with templates
   └── Identify sign type
        ↓
6. Temporal Filtering
   ├── Track across frames
   ├── Interpolation
   └── Smoothing
        ↓
7. Visualization
   └── Show metrics + sign type
```

## 🔧 Cấu Hình

### Bật/tắt tính năng:
```python
config.ENABLE_FEATURE_MATCHING = True
config.USE_COLOR_RATIO_ANALYSIS = True
config.USE_EDGE_DENSITY = True
config.USE_ASPECT_RATIO_FILTER = True
config.USE_TEXTURE_ANALYSIS = True
```

### Feature Detector Options:
```python
config.FEATURE_DETECTOR = 'SIFT'  # or 'ORB', 'AKAZE'
config.FEATURE_MATCH_THRESHOLD = 10  # Min good matches
config.MATCH_RATIO_THRESHOLD = 0.75  # Lowe's ratio
```

### Color Parameters (per color):
```python
'blue': {
    'min_color_ratio': 0.15,
    'expected_aspect_ratio': (0.8, 1.2),
    # ... existing params
}
```

## 📁 Chuẩn Bị Template Images

1. Tạo thư mục:
```bash
mkdir templates
```

2. Chuẩn bị ảnh template (rõ nét, góc nhìn thẳng):
   - `templates/stop_sign.jpg` - Biển dừng
   - `templates/no_entry.jpg` - Biển cấm vào
   - `templates/speed_limit.jpg` - Biển tốc độ
   - `templates/warning.jpg` - Biển cảnh báo

3. Cập nhật config:
```python
self.TEMPLATES = {
    'stop': 'templates/stop_sign.jpg',
    'no_entry': 'templates/no_entry.jpg',
    # ... thêm templates
}
```

## 🎨 Output Visualization

### Bounding Box Labels:
```
BLUE [stop] CR:0.35 Cir:0.95 FM:25
│    │      │        │        └─ Feature Matches
│    │      │        └─ Circularity
│    │      └─ Color Ratio
│    └─ Sign Type (từ feature matching)
└─ Color
```

### Debug Mode Info:
- Frame ID
- Number of detections
- ROI zones (colored rectangles)

## 📈 Performance Metrics

Mỗi detection sẽ có các metrics:
```python
{
    'color_ratio': 0.35,        # Tỷ lệ màu
    'aspect_ratio': 0.98,       # Width/Height
    'edge_density': 0.12,       # Mật độ cạnh
    'circularity': 0.95,        # Độ tròn (circles)
    'solidity': 0.88,           # Độ đặc (triangles)
    'feature_matches': 25,      # Số feature matches
    'match_confidence': 0.85,   # Confidence score
    'contrast': 0.65,           # Texture contrast
    'uniformity': 0.45,         # Texture uniformity
    'std_dev': 45.2,            # Độ phân tán
    'entropy': 5.8              # Entropy
}
```

## 🚀 Usage

### Chạy script mặc định:
```bash
python3 refactor_claude_enhanced.py
```

### Disable feature matching (nếu không có templates):
Sửa trong code:
```python
config.ENABLE_FEATURE_MATCHING = False
```

### Chỉ sử dụng color ratio và aspect ratio:
```python
config.USE_COLOR_RATIO_ANALYSIS = True
config.USE_ASPECT_RATIO_FILTER = True
config.USE_EDGE_DENSITY = False
config.USE_TEXTURE_ANALYSIS = False
config.ENABLE_FEATURE_MATCHING = False
```

## 🔍 Validation Logic

```python
def validate_detection(roi, bbox, color, params):
    # 1. Color Ratio Check
    if color_ratio < min_color_ratio:
        return False
    
    # 2. Aspect Ratio Check
    if not (min_ratio ≤ aspect_ratio ≤ max_ratio):
        return False
    
    # 3. Edge Density Check
    if edge_density < 0.05:
        return False
    
    # 4. Texture Check (optional)
    # ... texture analysis
    
    return True
```

## 💡 Tips

1. **Template Quality**: Sử dụng ảnh template rõ nét, góc nhìn thẳng
2. **Color Thresholds**: Điều chỉnh `min_color_ratio` theo từng loại biển
3. **Aspect Ratio**: Mở rộng range nếu biển báo bị méo góc nhìn
4. **Feature Detector**: 
   - SIFT: Chính xác nhất, chậm
   - ORB: Nhanh nhất, ít chính xác hơn
   - AKAZE: Cân bằng

## 📊 So Sánh với Phiên Bản Gốc

| Feature | Original | Enhanced |
|---------|----------|----------|
| Color Segmentation | ✓ | ✓ |
| Shape Detection | ✓ | ✓ |
| Temporal Filter | ✓ | ✓ |
| **Color Ratio Analysis** | ✗ | ✓ |
| **Aspect Ratio Filter** | ✗ | ✓ |
| **Edge Density** | ✗ | ✓ |
| **Texture Analysis** | ✗ | ✓ |
| **Feature Matching** | ✗ | ✓ |
| **Sign Type Recognition** | ✗ | ✓ |

## 🎓 Kỹ Thuật DIP Áp Dụng

1. **Color Space Transform**: BGR → HSV
2. **CLAHE**: Adaptive histogram equalization
3. **Morphological Operations**: Opening, Closing
4. **Edge Detection**: Canny algorithm
5. **Feature Detection**: SIFT/ORB/AKAZE
6. **Feature Matching**: Brute Force Matcher + Ratio Test
7. **Texture Analysis**: Histogram entropy, contrast
8. **Statistical Analysis**: Mean, std dev, ratios

## 🐛 Troubleshooting

**Không tìm thấy templates:**
```
⚠ Warning: Template directory 'templates' not found
   Feature matching will be disabled
```
→ Tạo folder `templates/` và thêm ảnh

**Quá nhiều false positives:**
→ Tăng `min_color_ratio` và `FEATURE_MATCH_THRESHOLD`

**Quá ít detections:**
→ Giảm thresholds, mở rộng `expected_aspect_ratio` range

**Chậm:**
→ Tắt feature matching hoặc dùng ORB thay vì SIFT
