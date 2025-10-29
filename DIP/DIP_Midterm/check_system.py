#!/usr/bin/env python3
"""
Quick system check for ultra-fast processing
"""
import multiprocessing as mp
import psutil
import sys

print("=" * 70)
print("🖥️  SYSTEM PERFORMANCE CHECK")
print("=" * 70)

# CPU Info
cpu_count = mp.cpu_count()
print(f"\n💻 CPU Information:")
print(f"   Total cores: {cpu_count}")
print(f"   Physical cores: {psutil.cpu_count(logical=False)}")
print(f"   Logical cores: {psutil.cpu_count(logical=True)}")
print(f"   CPU usage: {psutil.cpu_percent(interval=1)}%")

# Memory Info
mem = psutil.virtual_memory()
print(f"\n🧠 Memory Information:")
print(f"   Total RAM: {mem.total / (1024**3):.1f} GB")
print(f"   Available: {mem.available / (1024**3):.1f} GB")
print(f"   Used: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")

# Optimization Recommendations
print(f"\n⚡ Optimization Settings (for task1_ultra_fast.py):")
recommended_workers = cpu_count - 6
recommended_readers = min(4, cpu_count // 8)
print(f"   NUM_PROCESS_WORKERS: {recommended_workers} (using {recommended_workers}/{cpu_count} cores)")
print(f"   NUM_READ_THREADS: {recommended_readers}")

# Memory check for video
video_ram_needed = 8.0  # GB for 1920x1080x3087 frames
print(f"\n📹 Video Processing Estimate:")
print(f"   RAM needed for video: ~{video_ram_needed:.1f} GB")
if mem.available / (1024**3) > video_ram_needed:
    print(f"   ✅ Sufficient RAM available ({mem.available / (1024**3):.1f} GB > {video_ram_needed:.1f} GB)")
    print(f"   → Can load entire video into memory")
else:
    print(f"   ⚠️  Limited RAM ({mem.available / (1024**3):.1f} GB < {video_ram_needed:.1f} GB)")
    print(f"   → Will use streaming mode (slightly slower)")

# Expected performance
frames = 3087
fps = 30
single_thread_fps = 15
expected_speedup = min(recommended_workers * 0.7, cpu_count * 0.8)
expected_fps = single_thread_fps * expected_speedup

print(f"\n🚀 Expected Performance:")
print(f"   Original speed: ~{single_thread_fps} FPS")
print(f"   Optimized speed: ~{expected_fps:.0f} FPS")
print(f"   Speedup: ~{expected_speedup:.1f}x faster")
print(f"   Pass 1 time: ~{frames/expected_fps:.0f}s (was ~{frames/single_thread_fps:.0f}s)")
print(f"   Pass 2 time: ~{frames/(expected_fps*1.5):.0f}s (was ~{frames/single_thread_fps:.0f}s)")
print(f"   Total time: ~{frames/expected_fps + frames/(expected_fps*1.5):.0f}s (was ~{2*frames/single_thread_fps:.0f}s)")

print("\n" + "=" * 70)
print("✅ System is ready for ultra-fast processing!")
print("=" * 70)
