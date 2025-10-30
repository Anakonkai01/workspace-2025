## Quick context for AI coding agents

This repository contains several variants of the same DIP (traffic sign detection) solution for a single midterm assignment. The codebase is small and script-based (no package layout). Be conservative when editing: maintain the existing naming conventions and performance-focused choices.

Key files to reference
- `task_1_final_claude.py` — canonical two-pass implementation. PASS 1 detects and pickles all candidates; PASS 2 applies `TemporalSignFilter` and writes `detected_signs_filtered.mp4`.
- `task1_ultra_fast.py` — optimized/parallel variant. Auto-detects `CPU_CORES`, exposes runtime knobs (`NUM_PROCESS_WORKERS`, `NUM_READ_THREADS`, `FRAME_BUFFER_SIZE`, `BATCH_SIZE`). Uses ThreadPoolExecutor / ProcessPoolExecutor and multiprocessing.
- `task1_refine.py` / `task1_refine_super_performance.py` — intermediate/experimental variants (follow naming pattern `task1_*`).
- `check_system.py` — local system check and recommended worker/read settings (use before benchmarking).
- `PERFORMANCE_README.md` — performance notes (read for past measurements and rationale).

Architecture & data flow (concise)
- Single script entry points: each `task1_*.py` implements detection pipeline (frame preprocess → color masks → `extract_*_detections` → temporal filtering).
- Two-pass logical flow (used in `task_1_final_claude.py`):
  1. PASS 1: scan video, save all detections to a temp pickle (e.g. `temp_all_detections.pkl`).
  2. PASS 2: build `TemporalSignFilter` cache, filter short/flicker detections, write final annotated video.
- Core building blocks to reuse or modify:
  - `preprocess_frame(frame, clip_limit, blur_ksize)` — CLAHE + blur + HSV conversion.
  - `morphology(mask, k_size, iter_open, iter_close)` — morphological cleanup.
  - `extract_circle_detections(...)`, `extract_triangle_detections(...)` — contour-based logic with area/circularity/solidity heuristics.
  - `TemporalSignFilter` — IoU matching, track interpolation, smoothing, and cache building.

Project-specific conventions & patterns
- File naming: variants begin with `task1` or `task_1_`; preserve this when adding new runner scripts.
- Parameter tuning is done via global constants at top of scripts (color thresholds, CLAHE clip limits, kernel sizes). Change these there rather than adding hidden config files.
- Output artifacts: the code expects `task1.mp4` by default. PASS 1 writes `temp_all_detections.pkl` and PASS 2 writes `detected_signs_filtered.mp4`.
- Performance-first edits: `task1_ultra_fast.py` contains many optimizations (caching, multi-process pools). If changing core loops, keep an eye on shared-state and serialization costs.

Developer workflows (how to run locally)
- Quick system check (see recommended worker/readers):
```bash
python3 check_system.py
```
- Install minimal dependencies (if missing):
```bash
pip3 install opencv-python-headless numpy psutil
```
- Run canonical two-pass pipeline:
```bash
python3 task_1_final_claude.py
```
- Run the ultra-fast pipeline (multi-core):
```bash
python3 task1_ultra_fast.py
```

Editing guidance for AI agents
- Small, local edits only: change global constants or add small helper functions. Avoid large refactors across multiple `task1_*` variants—maintain behavior parity unless explicitly asked to consolidate.
- When modifying the detection thresholds, point to the specific global variables in the top of the target script (e.g., `blue_clahe_clip_limit`, `red_blur_ksize`).
- If adding dependencies, prefer lightweight, widely-available packages and update the README or add a `requirements.txt`.
- Preserve the two-pass contract: PASS 1 must produce a detection list (pickle) and PASS 2 must read/validate it and produce the final video. Tests and downstream scripts expect these artifacts.

Common pitfalls to avoid
- Changing parallelism without running `check_system.py` recommendations — may cause CPU oversubscription or OOM.
- Replacing NumPy/OpenCV operations with Python loops — performance regressions are easy and subtle.
- Renaming global variables (color thresholds, clip limits) without updating all `task1_*` variants.

If you need clarification or want a different level of change (refactor vs patch), tell me which script(s) to modify and whether to prioritise speed, clarity, or testability.

---
If anything in the instructions is unclear or you'd like extra examples (e.g., a small test harness or a requirements file), tell me which part to expand and I'll update the file.
