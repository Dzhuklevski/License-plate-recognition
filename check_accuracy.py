"""
Quick Accuracy Checker
======================
Quickly check the success rate of any approach

Usage:
    python check_accuracy.py results/predictions/approach1_results.json
    python check_accuracy.py results/predictions/approach2_results.json
    python check_accuracy.py results/predictions/approach3_results.json
"""

import json
import sys
from pathlib import Path

def check_accuracy(json_file):
    """Check success rate from results JSON"""

    if not Path(json_file).exists():
        print(f"❌ File not found: {json_file}")
        return

    with open(json_file, 'r') as f:
        data = json.load(f)

    total = len(data)
    successful = sum(1 for d in data if d.get('success', False))
    failed = total - successful
    success_rate = (successful / total * 100) if total > 0 else 0

    # Calculate average time if available
    times = [d.get('processing_time', 0) for d in data if 'processing_time' in d]
    avg_time = sum(times) / len(times) if times else 0
    fps = 1 / avg_time if avg_time > 0 else 0

    print("\n" + "="*60)
    print(f"RESULTS: {Path(json_file).name}")
    print("="*60)
    print(f"Total images:     {total}")
    print(f"Successful:       {successful} ({success_rate:.1f}%)")
    print(f"Failed:           {failed} ({(failed/total*100):.1f}%)")

    if avg_time > 0:
        print(f"Avg time:         {avg_time:.3f}s")
        print(f"Speed:            {fps:.1f} FPS")

    print("\n" + "="*60)

    # Show some examples
    print("\nSample successful detections:")
    success_samples = [d for d in data if d.get('success', False)][:5]
    for s in success_samples:
        print(f"  ✓ {s['filename']}: {s.get('detected_text', 'N/A')}")

    print("\nSample failures:")
    fail_samples = [d for d in data if not d.get('success', False)][:5]
    for f in fail_samples:
        print(f"  ✗ {f['filename']}")

    print("="*60 + "\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_accuracy.py <path_to_results.json>")
        print("\nExample:")
        print("  python check_accuracy.py results/predictions/approach1_results.json")
        sys.exit(1)

    check_accuracy(sys.argv[1])