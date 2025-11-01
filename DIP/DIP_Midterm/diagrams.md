# Traffic Sign Detection System Diagrams

## Block Diagram - System Architecture

```mermaid
graph TB
    subgraph "Input"
        A[Input Video<br/>task1.mp4]
    end
    
    subgraph "Global Parameters"
        B1[Color Parameters<br/>Blue/Red/Yellow<br/>HSV Ranges]
        B2[Area Parameters<br/>Circle/Triangle Thresholds]
        B3[Shape Quality<br/>Circularity/Solidity]
        B4[ROI Parameters<br/>Spatial Constraints]
        B5[Temporal Parameters<br/>Duration/Gap/IoU]
        B6[Performance Parameters<br/>CPU Cores/Workers/Threads]
    end
    
    subgraph "Detection Phase"
        C1[Frame Reader<br/>Read up to MAX_FRAME_ID]
        C2[Batch Creator<br/>Split into BATCH_SIZE]
        C3[Multi-Process Pool<br/>NUM_PROCESS_WORKERS]
        
        subgraph "Frame Processing"
            D1[Preprocess Frame<br/>CLAHE + Blur + HSV]
            D2[Color Masking<br/>Blue/Red/Yellow]
            D3[Morphology<br/>Open + Close]
            D4[Shape Detection<br/>Circles + Triangles]
            D5[ROI Filtering<br/>Two-Layer Logic]
        end
        
        C1 --> C2
        C2 --> C3
        C3 --> D1
        D1 --> D2
        D2 --> D3
        D3 --> D4
        D4 --> D5
    end
    
    subgraph "Temporal Filtering"
        E1[TemporalSignFilter]
        E2[Track Building<br/>IoU Matching]
        E3[Interpolation<br/>Fill Missing Frames]
        E4[Smoothing<br/>Moving Average]
        E5[Validation<br/>Min Duration Filter]
        E6[Detection Cache<br/>Frame → Detections Map]
        
        E1 --> E2
        E2 --> E3
        E3 --> E4
        E4 --> E5
        E5 --> E6
    end
    
    subgraph "Rendering Phase"
        F1[Multi-Thread Pool<br/>Readers + Processors]
        F2[Frame Reader Threads<br/>NUM_READ_THREADS]
        F3[Frame Processor Threads<br/>NUM_PROCESS_WORKERS]
        
        subgraph "Frame Rendering"
            G1[Get Validated Detections<br/>From Cache]
            G2[Draw Frame ID<br/>if DEBUG_MODE]
            G3[Draw ROI Boxes<br/>if DEBUG_MODE]
            G4[Draw Detections<br/>Bboxes + Metrics]
        end
        
        F1 --> F2
        F1 --> F3
        F2 --> G1
        F3 --> G1
        G1 --> G2
        G2 --> G3
        G3 --> G4
    end
    
    subgraph "Output"
        H[Output Video<br/>task1_output.mp4]
    end
    
    A --> C1
    B1 & B2 & B3 & B4 & B5 --> D1
    B5 --> E1
    B6 --> C3
    B6 --> F1
    D5 --> E1
    E6 --> G1
    G4 --> H
    
    style A fill:#e1f5ff
    style H fill:#d4edda
    style C3 fill:#fff3cd
    style F1 fill:#fff3cd
    style E6 fill:#f8d7da
```

## Data Flow Diagram - Processing Pipeline

```mermaid
flowchart TD
    Start([Start]) --> LoadVideo[Load Video<br/>Get fps, width, height, total_frames]
    
    LoadVideo --> InitParams[Initialize Parameters<br/>- Color thresholds<br/>- Area limits<br/>- ROI regions<br/>- Temporal params]
    
    InitParams --> CreateFilter[Create TemporalSignFilter<br/>with color-specific params]
    
    CreateFilter --> DetectionPhase{Detection Phase}
    
    subgraph " "
        DetectionPhase --> ReadFrames[Read Frames 0 to MAX_FRAME_ID]
        ReadFrames --> SplitBatches[Split into Batches<br/>size = BATCH_SIZE]
        
        SplitBatches --> ParallelProcess[Parallel Processing<br/>NUM_PROCESS_WORKERS processes]
        
        ParallelProcess --> ProcessBatch[For Each Batch]
        
        ProcessBatch --> CropFrame[Crop Frame<br/>height_new = h × 0.475]
        
        CropFrame --> BlueDetect[Blue Channel:<br/>Preprocess → Mask → Morphology<br/>→ Extract Circles → ROI Filter]
        CropFrame --> RedDetect[Red Channel:<br/>Preprocess → Mask → Morphology<br/>→ Extract Circles → ROI Filter]
        CropFrame --> YellowDetect[Yellow Channel:<br/>Preprocess → Mask → Morphology<br/>→ Extract Triangles → ROI Filter]
        
        BlueDetect --> CombineDetections[Combine All Detections<br/>with metrics: area, circularity, solidity]
        RedDetect --> CombineDetections
        YellowDetect --> CombineDetections
        
        CombineDetections --> AddToFilter[Add to TemporalSignFilter<br/>frame_num, detections]
    end
    
    AddToFilter --> BuildTracks[Build Temporal Tracks<br/>- IoU matching<br/>- Gap filling<br/>- Color-specific thresholds]
    
    BuildTracks --> InterpolateGaps[Interpolate Missing Frames<br/>Linear interpolation of bbox]
    
    InterpolateGaps --> SmoothTracks[Smooth Bounding Boxes<br/>Moving average window=5]
    
    SmoothTracks --> ValidateTracks[Validate Tracks<br/>Filter by min duration<br/>color-specific]
    
    ValidateTracks --> BuildCache[Build Detection Cache<br/>Map: frame_num → validated_detections]
    
    BuildCache --> Stats[Get Statistics<br/>total_tracks, valid_tracks]
    
    Stats --> RenderingPhase{Rendering Phase}
    
    subgraph "  "
        RenderingPhase --> ReadAllFrames[Read ALL Frames 0 to total_frames]
        
        ReadAllFrames --> MultiThread[Multi-Threading<br/>NUM_READ_THREADS readers<br/>NUM_PROCESS_WORKERS processors]
        
        MultiThread --> ReadQueue[Reader Threads<br/>→ Read Queue]
        MultiThread --> ProcessQueue[Processor Threads<br/>→ Process Queue]
        
        ReadQueue --> GetFromCache[Get Validated Detections<br/>from Cache for frame_num]
        ProcessQueue --> GetFromCache
        
        GetFromCache --> CheckDebug{DEBUG_MODE?}
        
        CheckDebug -->|Yes| DrawFrameID[Draw Frame ID]
        CheckDebug -->|Yes| DrawROI[Draw ROI Boxes<br/>Blue/Red/Yellow]
        CheckDebug -->|No| DrawDetections[Draw Detections Only]
        
        DrawFrameID --> DrawROI
        DrawROI --> DrawDetectionsDebug[Draw Detections<br/>+ Area + Circularity/Solidity]
        
        DrawDetectionsDebug --> WriteFrame[Write Frame to Output]
        DrawDetections --> WriteFrame
        
        WriteFrame --> FrameBuffer[Frame Buffer<br/>Reorder if needed]
    end
    
    FrameBuffer --> SaveVideo[Save Output Video<br/>task1_output.mp4]
    
    SaveVideo --> PrintStats[Print Statistics<br/>- Processing time<br/>- FPS<br/>- Track retention]
    
    PrintStats --> End([End])
    
    style Start fill:#d4edda
    style End fill:#d4edda
    style DetectionPhase fill:#fff3cd
    style RenderingPhase fill:#fff3cd
    style BuildCache fill:#f8d7da
    style ParallelProcess fill:#ffeaa7
    style MultiThread fill:#ffeaa7
```

## Sequence Diagram - Two-Phase Processing

```mermaid
sequenceDiagram
    actor User
    participant Main
    participant VideoReader
    participant ProcessPool as Process Pool<br/>(Workers)
    participant TempFilter as Temporal Filter
    participant Cache
    participant ThreadPool as Thread Pool<br/>(Readers+Processors)
    participant VideoWriter
    
    User->>Main: Run program
    Main->>VideoReader: Open task1.mp4
    VideoReader-->>Main: Return fps, dimensions, frame_count
    Main->>TempFilter: Initialize with color_params
    
    rect rgb(255, 243, 205)
        Note over Main,TempFilter: DETECTION PHASE
        Main->>VideoReader: Read frames 0 to MAX_FRAME_ID
        VideoReader-->>Main: Return frames
        Main->>Main: Split into batches (BATCH_SIZE)
        
        loop For each batch
            Main->>ProcessPool: Submit batch for processing
            ProcessPool->>ProcessPool: Crop + Preprocess + Detect
            ProcessPool-->>Main: Return detections with metrics
            Main->>TempFilter: Add detections (frame_num, detections)
            TempFilter->>TempFilter: Track building (IoU matching)
        end
        
        Main->>TempFilter: Build detection cache
        TempFilter->>TempFilter: Interpolate missing frames
        TempFilter->>TempFilter: Smooth bounding boxes
        TempFilter->>TempFilter: Validate by min duration
        TempFilter->>Cache: Store validated detections
        Cache-->>TempFilter: Cache built
        TempFilter-->>Main: Statistics (total_tracks, valid_tracks)
    end
    
    rect rgb(212, 237, 218)
        Note over Main,VideoWriter: RENDERING PHASE
        Main->>ThreadPool: Start reader + processor threads
        
        par Parallel Reading
            ThreadPool->>VideoReader: Read frames (multi-threaded)
            VideoReader-->>ThreadPool: Frame batches → Read Queue
        and Parallel Processing
            ThreadPool->>Cache: Get validated detections
            Cache-->>ThreadPool: Return detections for frame
            ThreadPool->>ThreadPool: Draw detections + ROI + Frame ID
            ThreadPool-->>ThreadPool: → Process Queue
        end
        
        loop For each frame (in order)
            ThreadPool->>VideoWriter: Write processed frame
        end
        
        ThreadPool-->>Main: Rendering complete
        Main->>VideoWriter: Close output video
    end
    
    Main-->>User: Display statistics and completion
```

## Class Diagram - TemporalSignFilter

```mermaid
classDiagram
    class TemporalSignFilter {
        -float fps
        -dict min_frames
        -dict max_gap_frames
        -dict iou_thresholds
        -defaultdict tracks
        -int next_track_id
        -dict _validated_cache
        -bool _cache_built
        
        +__init__(fps, color_params)
        +calculate_iou(box1, box2) float
        +add_detections(frame_num, detections)
        +interpolate_missing_frames(track_data, color) list
        +smooth_bounding_boxes(track_data, window_size) list
        +build_detection_cache()
        +get_validated_detections(frame_num) list
        +get_statistics() tuple
    }
    
    class Detection {
        +tuple bbox (x, y, w, h)
        +str color (blue/red/yellow)
        +dict metrics (area, circularity/solidity, shape)
    }
    
    class Track {
        +int frame
        +tuple bbox
        +str color
        +dict metrics
        +bool interpolated
    }
    
    TemporalSignFilter "1" --> "*" Track : manages
    Track "1" --> "1" Detection : contains
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph "Input Layer"
        A[Video File<br/>task1.mp4]
    end
    
    subgraph "Processing Layer"
        B1[Frame Preprocessing<br/>CLAHE + HSV]
        B2[Color Segmentation<br/>Thresholding]
        B3[Morphological Ops<br/>Open + Close]
        B4[Shape Detection<br/>Contours]
        B5[ROI Validation<br/>Two-Layer]
    end
    
    subgraph "Tracking Layer"
        C1[IoU Matching]
        C2[Track Building]
        C3[Interpolation]
        C4[Smoothing]
        C5[Validation]
    end
    
    subgraph "Cache Layer"
        D1[Detection Cache<br/>frame → detections]
    end
    
    subgraph "Rendering Layer"
        E1[Frame Reading]
        E2[Cache Lookup]
        E3[Drawing]
        E4[Video Writing]
    end
    
    subgraph "Output Layer"
        F[Output Video<br/>task1_output.mp4]
    end
    
    A --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> B5
    B5 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> C5
    C5 --> D1
    D1 --> E2
    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> F
    
    style A fill:#e1f5ff
    style D1 fill:#f8d7da
    style F fill:#d4edda
```

## Performance Optimization Flow

```mermaid
graph TD
    Start([System Start]) --> DetectCores[Detect CPU Cores]
    
    DetectCores --> CalcThreads[Calculate NUM_READ_THREADS<br/>max 1, min 4, CPU_CORES // 8]
    
    CalcThreads --> CalcWorkers[Calculate NUM_PROCESS_WORKERS<br/>CPU_CORES - NUM_READ_THREADS - 2]
    
    CalcWorkers --> CalcBuffer[Calculate FRAME_BUFFER_SIZE<br/>min 30, max 200, CPU_CORES × 4]
    
    CalcBuffer --> CalcBatch[Calculate BATCH_SIZE<br/>min 10, max 50, CPU_CORES]
    
    CalcBatch --> DetectionOpt{Detection Phase<br/>Optimization}
    
    DetectionOpt --> MultiProcess[Use ProcessPoolExecutor<br/>NUM_PROCESS_WORKERS processes]
    
    MultiProcess --> BatchProcess[Process frames in batches<br/>BATCH_SIZE frames/batch]
    
    BatchProcess --> RenderingOpt{Rendering Phase<br/>Optimization}
    
    RenderingOpt --> ReaderThreads[Spawn NUM_READ_THREADS<br/>reader threads]
    
    RenderingOpt --> ProcessorThreads[Spawn NUM_PROCESS_WORKERS<br/>processor threads]
    
    ReaderThreads --> ReadQueue[Read Queue<br/>size = FRAME_BUFFER_SIZE]
    ProcessorThreads --> ProcessQueue[Process Queue<br/>size = FRAME_BUFFER_SIZE]
    
    ReadQueue --> FrameBuffer[Frame Buffer<br/>Handles out-of-order frames]
    ProcessQueue --> FrameBuffer
    
    FrameBuffer --> OrderedWrite[Write frames in order<br/>Sequential video output]
    
    OrderedWrite --> Complete([Complete])
    
    style Start fill:#d4edda
    style Complete fill:#d4edda
    style MultiProcess fill:#fff3cd
    style ReaderThreads fill:#fff3cd
    style ProcessorThreads fill:#fff3cd
    style FrameBuffer fill:#f8d7da
```

