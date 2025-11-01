# Core Logic Diagrams - Traffic Sign Detection System

## 1. Two-Layer ROI Filtering Logic

```mermaida
%%{init: {'theme':'base', 'themeVariables': {'fontSize':'14px'}}}%%
graph TD
    A["Detected Shape<br/>area, bbox"] --> B["Is area >= TRUST_THRESHOLD?"]
    
    B -->|Yes| C["Trust Large Shapes<br/>Bypass ROI check<br/>Accept detection"]
    
    B -->|No| D["Is bbox inside ROI?<br/>overlap >= 50%"]
    
    D -->|Yes| E["Accept Small Shape<br/>in ROI"]
    
    D -->|No| F["Reject Shape<br/>Likely noise"]
    
    C --> G["✅ Valid Detection"]
    E --> G
    F --> H["❌ Filtered Out"]
    
    style A fill:#e1f5ff
    style G fill:#d4edda
    style H fill:#f8d7da
    style B fill:#fff3cd
    style D fill:#fff3cd
```

**Significance:** Balances sensitivity (small detections in ROI) with specificity (rejects distant noise)

---

## 2. Temporal Track Building - IoU Matching

```mermaid
graph TD
    A["New Detection<br/>frame_n, bbox, color"] --> B["Find matching track<br/>from previous frames"]
    
    B --> C["Filter candidates<br/>Same color?"]
    
    C -->|No| D["Create new track"]
    
    C -->|Yes| E["Check time gap<br/>frame_n - last_frame<br/><= MAX_GAP_SEC?"]
    
    E -->|No| F["Create new track"]
    
    E -->|Yes| G["Calculate IoU<br/>with last detection"]
    
    G --> H["IoU >= THRESHOLD?"]
    
    H -->|Yes| I["Link to existing track<br/>Extend track with new bbox"]
    
    H -->|No| J["Create new track"]
    
    D --> K["✅ Track assigned"]
    F --> K
    I --> K
    J --> K
    
    style A fill:#e1f5ff
    style K fill:#d4edda
    style B fill:#fff3cd
    style E fill:#fff3cd
    style H fill:#fff3cd
```

**Significance:** Color-specific matching prevents blue circles from being linked with red circles. Gap threshold handles sign occlusion. IoU ensures spatial continuity.

---

## 3. Shape Detection Pipeline - Blue/Red Circles

```mermaid
graph TD
    A["Input Mask<br/>binary image"] --> B["Find Contours<br/>cv2.findContours"]
    
    B --> C["Calculate Area<br/>cv2.contourArea"]
    
    C --> D["Area Filter<br/>MIN_AREA <= area <= MAX_AREA"]
    
    D -->|Fail| E["❌ Reject"]
    
    D -->|Pass| F["Calculate Hull<br/>cv2.convexHull"]
    
    F --> G["Calculate Circularity<br/>4π×(area/perimeter²)"]
    
    G --> H["Circularity Filter<br/>circularity >= THRESHOLD"]
    
    H -->|Fail| E
    
    H -->|Pass| I["Extract ROI check<br/>Two-layer logic"]
    
    I -->|Pass| J["✅ Circle Detection<br/>with metrics"]
    
    E --> K["❌ Filtered"]
    
    style A fill:#e1f5ff
    style J fill:#d4edda
    style K fill:#f8d7da
    style D fill:#fff3cd
    style H fill:#fff3cd
    style I fill:#fff3cd
```

**Significance:** 
- **Area filter**: Rejects noise (too small) and false positives (too large)
- **Circularity**: Distinguishes circles from irregular shapes
- **ROI + Trust threshold**: Handles both small precise detections and large confident detections

---

## 4. Shape Detection Pipeline - Yellow Triangles

```mermaid
graph TD
    A["Input Mask<br/>binary image"] --> B["Find Contours"]
    
    B --> C["Calculate Area<br/>& Hull"]
    
    C --> D["Area Filter<br/>MIN_AREA <= area <= MAX_AREA"]
    
    D -->|Fail| E["❌ Reject"]
    
    D -->|Pass| F["Calculate Solidity<br/>area / hull_area"]
    
    F --> G["Solidity Filter<br/>solidity >= MIN_SOLIDITY"]
    
    G -->|Fail| E
    
    G -->|Pass| H["Approximate Polygon<br/>cv2.approxPolyDP<br/>epsilon ∝ perimeter"]
    
    H --> I["Vertex Count Filter<br/>vertices <= MAX_VERTICES"]
    
    I -->|Fail| E
    
    I -->|Pass| J["Extract ROI check<br/>Two-layer logic"]
    
    J -->|Pass| K["✅ Triangle Detection<br/>with metrics"]
    
    style A fill:#e1f5ff
    style K fill:#d4edda
    style E fill:#f8d7da
    style D fill:#fff3cd
    style G fill:#fff3cd
    style I fill:#fff3cd
    style J fill:#fff3cd
```

**Significance:**
- **Solidity**: Ensures compact triangular shapes (rejects thin/elongated contours)
- **Polygon approximation**: Counts vertices to confirm triangular structure (3-7 vertices allowed)
- **Epsilon proportional to perimeter**: Adapts to different contour sizes

---

## 5. Track Validation & Filtering

```mermaid
graph TD
    A["Built Tracks<br/>from all detections"] --> B["For each track:<br/>Get color from first detection"]
    
    B --> C["Get color-specific<br/>MIN_DURATION_SEC"]
    
    C --> D["Calculate track duration<br/>last_frame - first_frame"]
    
    D --> E["Duration >= MIN_DURATION?"]
    
    E -->|No| F["❌ Invalid Track<br/>Too short - likely noise"]
    
    E -->|Yes| G["✅ Valid Track<br/>Add to cache"]
    
    F --> H["Track Filtering<br/>Statistics"]
    
    G --> H
    
    H --> I["retention_rate =<br/>valid_tracks / total_tracks"]
    
    style A fill:#e1f5ff
    style G fill:#d4edda
    style F fill:#f8d7da
    style E fill:#fff3cd
```

**Significance:**
- **Color-specific validation**: Blue/Red signs have different temporal patterns than yellow
- **Min duration threshold**: Filters jittering/flickering false positives
- **Retention rate**: Indicates detection quality

---

## 6. Interpolation & Smoothing Strategy

```mermaid
graph LR
    A["Detected Track<br/>Frame 10, 12, 15"] 
    
    B["Interpolate<br/>Fill Frame 11, 13-14<br/>Linear bbox interpolation<br/>max_gap = 0.5s color-specific"]
    
    C["After Interpolation<br/>Frame 10-15 continuous<br/>with intermediate bboxes"]
    
    D["Smooth<br/>Moving average<br/>window = 5<br/>Center-aligned"]
    
    E["Final Track<br/>Smooth + complete<br/>10-15 frames"]
    
    A --> B
    B --> C
    C --> D
    D --> E
    
    style A fill:#e1f5ff
    style E fill:#d4edda
    style B fill:#fff3cd
    style D fill:#fff3cd
```

**Significance:**
- **Interpolation**: Creates seamless tracks despite detection gaps
- **Color-specific max_gap**: Yellow can tolerate longer gaps than blue/red
- **Smoothing**: Reduces jitter in bbox coordinates for stable rendering

---

## 7. Frame Preprocessing Impact

```mermaid
graph TD
    A["Original Frame"] --> B["Median Blur<br/>kernel = color-specific<br/>Removes salt-and-pepper noise"]
    
    B --> C["Convert to HSV<br/>Better color separation<br/>than RGB"]
    
    C --> D["Split H, S, V channels"]
    
    D --> E["CLAHE on V channel<br/>clipLimit = param/10<br/>Enhances local contrast"]
    
    D --> F["Saturation Boost<br/>S * 1.5<br/>Emphasizes color"]
    
    E --> G["Merge back<br/>Enhanced HSV"]
    
    F --> G
    
    G --> H["Color Thresholding<br/>Now much more robust<br/>to lighting variations"]
    
    style A fill:#e1f5ff
    style H fill:#d4edda
    style B fill:#fff3cd
    style C fill:#fff3cd
    style E fill:#fff3cd
    style F fill:#fff3cd
```

**Significance:**
- **Median blur**: Noise reduction without blurring edges
- **HSV space**: More robust to shadows/lighting than RGB
- **CLAHE**: Adaptive histogram equalization - handles dark/bright regions
- **Saturation boost**: Makes colors pop for better thresholding

---

## 8. Color-Specific Parameters Impact - Version 1

```mermaid
graph TD
    subgraph Blue["BLUE Circles"]
        B1["HSV: 102-144°, 216-255S, 81-227V"]
        B2["Small blur (7px)"]
        B3["Light morphology"]
        B4["Min: 2.0s, Gap: 0.5s"]
    end
    
    subgraph Red["RED Circles"]
        R1["HSV: 117-179°, 40-255S, 0-255V"]
        R2["Tiny blur (5px)"]
        R3["Moderate morphology"]
        R4["Min: 2.0s, Gap: 0.5s"]
    end
    
    subgraph Yellow["YELLOW Triangles"]
        Y1["HSV: 8-18°, 111-255S, 100-255V"]
        Y2["Medium blur (7px)"]
        Y3["Light morphology"]
        Y4["Min: 3.0s, Gap: 0.5s"]
    end
    
    B1 --> B2 --> B3 --> B4
    R1 --> R2 --> R3 --> R4
    Y1 --> Y2 --> Y3 --> Y4
    
    style Blue fill:#e1f5ff
    style Red fill:#fce4ec
    style Yellow fill:#fffde7
```

**Significance:**

- Different colors need different processing
- Red has wider V range (handles both dark and bright red)
- Yellow tolerates longer gaps (may disappear longer in traffic)
- Morphology parameters tuned per color

---

## 8B. Color-Specific Parameters Impact - Version 2 (Detailed Comparison)

```mermaid
graph LR
    Start["Color Parameters"] --> Blue["BLUE"]
    Start --> Red["RED"]
    Start --> Yellow["YELLOW"]
    
    Blue --> B_HSV["HSV Range<br/>102-144° Hue<br/>216-255 Saturation<br/>81-227 Value"]
    Blue --> B_Shape["Shape: Circle"]
    Blue --> B_Blur["Blur: 7px<br/>Medium blur"]
    Blue --> B_Morph["Morphology<br/>Open: 1 iter<br/>Close: 5 iter"]
    Blue --> B_Tempo["Temporal<br/>Min: 2.0s<br/>Gap: 0.5s<br/>IoU: 0.3"]
    
    Red --> R_HSV["HSV Range<br/>117-179° Hue<br/>40-255 Saturation<br/>0-255 Value"]
    Red --> R_Shape["Shape: Circle"]
    Red --> R_Blur["Blur: 5px<br/>Small blur"]
    Red --> R_Morph["Morphology<br/>Open: 2 iter<br/>Close: 5 iter"]
    Red --> R_Tempo["Temporal<br/>Min: 2.0s<br/>Gap: 0.5s<br/>IoU: 0.3"]
    
    Yellow --> Y_HSV["HSV Range<br/>8-18° Hue<br/>111-255 Saturation<br/>100-255 Value"]
    Yellow --> Y_Shape["Shape: Triangle"]
    Yellow --> Y_Blur["Blur: 7px<br/>Medium blur"]
    Yellow --> Y_Morph["Morphology<br/>Open: 1 iter<br/>Close: 5 iter"]
    Yellow --> Y_Tempo["Temporal<br/>Min: 3.0s<br/>Gap: 0.5s<br/>IoU: 0.3"]
    
    B_HSV --> B_Result["✅ Optimized<br/>for BLUE"]
    B_Shape --> B_Result
    B_Blur --> B_Result
    B_Morph --> B_Result
    B_Tempo --> B_Result
    
    R_HSV --> R_Result["✅ Optimized<br/>for RED"]
    R_Shape --> R_Result
    R_Blur --> R_Result
    R_Morph --> R_Result
    R_Tempo --> R_Result
    
    Y_HSV --> Y_Result["✅ Optimized<br/>for YELLOW"]
    Y_Shape --> Y_Result
    Y_Blur --> Y_Result
    Y_Morph --> Y_Result
    Y_Tempo --> Y_Result
    
    style Blue fill:#e1f5ff
    style Red fill:#fce4ec
    style Yellow fill:#fffde7
    style B_Result fill:#c8e6c9
    style R_Result fill:#c8e6c9
    style Y_Result fill:#c8e6c9
```

**Why Different Parameters:**

- **Blue**: Narrow V range (81-227) - consistent blue signs
- **Red**: Wider V range (0-255) - red varies light/dark
- **Yellow**: Higher min V (100) - yellow needs brightness
- **Yellow**: 3.0s min (longer than others) - yellow signs stay visible longer
- **Red**: Smaller blur (5px) - preserves fine red details
- **Circles vs Triangles**: Different shape validation criteria

---

## 9. End-to-End Quality Pipeline

```mermaid
graph TD
    A["Raw Frame"] 
    
    A --> B["Stage 1:<br/>Preprocessing<br/>CLAHE + Saturation<br/>↓ Noise, ↑ Contrast"]
    
    B --> C["Stage 2:<br/>Shape Detection<br/>Area + Circularity/Solidity<br/>↓ False positives"]
    
    C --> D["Stage 3:<br/>ROI Filtering<br/>Two-layer logic<br/>↓ Distant noise"]
    
    D --> E["Stage 4:<br/>Temporal Tracking<br/>IoU matching<br/>↓ Jitter"]
    
    E --> F["Stage 5:<br/>Track Validation<br/>Min duration<br/>↓ Flickers"]
    
    F --> G["Stage 6:<br/>Interpolation<br/>Linear bbox<br/>↑ Completeness"]
    
    G --> H["Stage 7:<br/>Smoothing<br/>Moving average<br/>↓ Vibration"]
    
    H --> I["Final Output<br/>High-quality tracks<br/>Ready for rendering"]
    
    style A fill:#e1f5ff
    style I fill:#d4edda
    style B fill:#fff3cd
    style C fill:#fff3cd
    style D fill:#fff3cd
    style E fill:#fff3cd
    style F fill:#fff3cd
    style G fill:#fff3cd
    style H fill:#fff3cd
```

**Significance:** Each stage filters out specific types of errors. The cascade approach is more robust than single-stage filtering.

---

## Summary - Key Logic Components

| Component | Purpose | Impact |
|-----------|---------|--------|
| **Two-Layer ROI** | Balance sensitivity/specificity | Reduces false positives while keeping real detections |
| **IoU Matching** | Track consistency | Links same sign across frames |
| **Color-Specific Params** | Adapt to sign properties | Each sign type detected optimally |
| **Interpolation** | Handle occlusions | Smooth tracks despite gaps |
| **Smoothing** | Reduce jitter | Stable bbox for rendering |
| **Validation** | Confirm real signs | Filters out noise and flickers |
| **CLAHE + Saturation** | Robust preprocessing | Works in varied lighting |
| **Shape metrics** | Distinguish objects | Circularity for circles, Solidity for triangles |
