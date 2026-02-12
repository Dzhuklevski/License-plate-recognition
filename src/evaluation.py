"""
EVALUATION AND COMPARISON
=========================

This script evaluates all three approaches and compares their performance.

Metrics:
- Detection Accuracy (IoU > 0.5)
- Character Recognition Accuracy
- Processing Speed (FPS)
- Success Rate

Usage:
    python src/evaluation.py --test_dir data/processed/splits/test
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union
    
    Args:
        box1, box2: (x, y, w, h) format
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (x1, y1, x2, y2)
    box1_coords = [x1, y1, x1 + w1, y1 + h1]
    box2_coords = [x2, y2, x2 + w2, y2 + h2]
    
    # Intersection
    x_left = max(box1_coords[0], box2_coords[0])
    y_top = max(box1_coords[1], box2_coords[1])
    x_right = min(box1_coords[2], box2_coords[2])
    y_bottom = min(box1_coords[3], box2_coords[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0


def load_ground_truth(annotations_dir):
    """
    Load ground truth bounding boxes from annotations
    """
    import xml.etree.ElementTree as ET
    
    annotations_dir = Path(annotations_dir)
    ground_truth = {}
    
    for xml_file in annotations_dir.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find('filename').text
        bbox = root.find('object/bndbox')
        
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        ground_truth[filename] = {
            'bbox': (xmin, ymin, xmax - xmin, ymax - ymin)
        }
    
    return ground_truth


def evaluate_detection(predictions, ground_truth, iou_threshold=0.5):
    """
    Evaluate detection accuracy using IoU
    """
    correct = 0
    total = 0
    ious = []
    
    for filename, pred in predictions.items():
        if filename not in ground_truth:
            continue
        
        total += 1
        
        if pred['bbox'] is None:
            ious.append(0)
            continue
        
        gt_bbox = ground_truth[filename]['bbox']
        pred_bbox = pred['bbox']
        
        iou = calculate_iou(pred_bbox, gt_bbox)
        ious.append(iou)
        
        if iou >= iou_threshold:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    mean_iou = np.mean(ious) if ious else 0
    
    return {
        'detection_accuracy': accuracy,
        'mean_iou': mean_iou,
        'correct': correct,
        'total': total
    }


def calculate_character_accuracy(pred_text, gt_text):
    """
    Calculate character-level accuracy
    """
    if not gt_text:
        return 0.0
    
    if not pred_text:
        return 0.0
    
    # Make same length for comparison
    max_len = max(len(pred_text), len(gt_text))
    pred_padded = pred_text.ljust(max_len, ' ')
    gt_padded = gt_text.ljust(max_len, ' ')
    
    correct = sum(1 for p, g in zip(pred_padded, gt_padded) if p == g)
    
    return correct / max_len


def load_predictions(json_file):
    """
    Load predictions from JSON file
    """
    with open(json_file, 'r') as f:
        predictions = json.load(f)
    
    # Convert to dictionary
    pred_dict = {}
    for pred in predictions:
        filename = pred['filename']
        pred_dict[filename] = {
            'text': pred.get('detected_text', ''),
            'bbox': pred.get('bbox'),
            'confidence': pred.get('confidence', 0),
            'success': pred.get('success', False),
            'time': pred.get('processing_time', 0)
        }
    
    return pred_dict


class PerformanceEvaluator:
    """
    Comprehensive performance evaluation
    """
    
    def __init__(self, test_dir):
        self.test_dir = Path(test_dir)
        self.annotations_dir = self.test_dir.parent / 'annotations'
        
        # Load ground truth
        print("Loading ground truth annotations...")
        self.ground_truth = load_ground_truth(self.annotations_dir)
        print(f"Loaded {len(self.ground_truth)} annotations")
    
    def evaluate_approach(self, predictions_file, approach_name):
        """
        Evaluate single approach
        """
        print(f"\nEvaluating {approach_name}...")
        
        predictions = load_predictions(predictions_file)
        
        # Detection metrics
        det_metrics = evaluate_detection(predictions, self.ground_truth)
        
        # Speed metrics
        times = [p['time'] for p in predictions.values() if p['time'] > 0]
        avg_time = np.mean(times) if times else 0
        fps = 1 / avg_time if avg_time > 0 else 0
        
        # Success rate
        total_pred = len(predictions)
        successful = sum(1 for p in predictions.values() if p['success'])
        success_rate = successful / total_pred if total_pred > 0 else 0
        
        results = {
            'approach': approach_name,
            'detection_accuracy': det_metrics['detection_accuracy'] * 100,
            'mean_iou': det_metrics['mean_iou'],
            'success_rate': success_rate * 100,
            'avg_time': avg_time,
            'fps': fps,
            'total_images': total_pred
        }
        
        return results
    
    def compare_approaches(self, results_files):
        """
        Compare all approaches
        """
        all_results = []
        
        for approach_name, file_path in results_files.items():
            if not Path(file_path).exists():
                print(f"⚠ {file_path} not found, skipping {approach_name}")
                continue
            
            results = self.evaluate_approach(file_path, approach_name)
            all_results.append(results)
        
        return pd.DataFrame(all_results)
    
    def create_comparison_report(self, df, output_dir):
        """
        Create visual comparison report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Detection Accuracy
        ax = axes[0, 0]
        df.plot(x='approach', y='detection_accuracy', kind='bar', ax=ax, legend=False)
        ax.set_title('Detection Accuracy (IoU > 0.5)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('')
        ax.set_ylim([0, 100])
        for i, v in enumerate(df['detection_accuracy']):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # 2. Success Rate
        ax = axes[0, 1]
        df.plot(x='approach', y='success_rate', kind='bar', ax=ax, legend=False, color='green')
        ax.set_title('Overall Success Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate (%)')
        ax.set_xlabel('')
        ax.set_ylim([0, 100])
        for i, v in enumerate(df['success_rate']):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # 3. Processing Speed
        ax = axes[1, 0]
        df.plot(x='approach', y='fps', kind='bar', ax=ax, legend=False, color='orange')
        ax.set_title('Processing Speed', fontsize=12, fontweight='bold')
        ax.set_ylabel('FPS (Frames Per Second)')
        ax.set_xlabel('')
        for i, v in enumerate(df['fps']):
            ax.text(i, v + 0.5, f'{v:.1f}', ha='center', fontweight='bold')
        
        # 4. Mean IoU
        ax = axes[1, 1]
        df.plot(x='approach', y='mean_iou', kind='bar', ax=ax, legend=False, color='red')
        ax.set_title('Mean IoU Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('IoU')
        ax.set_xlabel('')
        ax.set_ylim([0, 1])
        for i, v in enumerate(df['mean_iou']):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        plot_path = output_dir / 'comparison_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Comparison plots saved: {plot_path}")
        plt.close()
        
        # Create summary table
        self.create_summary_table(df, output_dir)
    
    def create_summary_table(self, df, output_dir):
        """
        Create text summary table
        """
        summary_file = output_dir / 'comparison_summary.txt'

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LICENSE PLATE DETECTION & RECOGNITION - COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-"*80 + "\n\n")
            
            # Format table
            f.write(f"{'Approach':<25} {'Det.Acc':<12} {'Success':<12} {'FPS':<12} {'Mean IoU':<12}\n")
            f.write("-"*80 + "\n")
            
            for _, row in df.iterrows():
                f.write(f"{row['approach']:<25} "
                       f"{row['detection_accuracy']:>10.1f}%  "
                       f"{row['success_rate']:>10.1f}%  "
                       f"{row['fps']:>10.1f}  "
                       f"{row['mean_iou']:>10.3f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("\nKEY FINDINGS:\n")
            f.write("-"*80 + "\n")
            
            # Best in each category
            best_detection = df.loc[df['detection_accuracy'].idxmax()]
            best_speed = df.loc[df['fps'].idxmax()]
            best_iou = df.loc[df['mean_iou'].idxmax()]
            
            f.write(f"\n✓ Best Detection Accuracy: {best_detection['approach']} ({best_detection['detection_accuracy']:.1f}%)\n")
            f.write(f"✓ Fastest Processing:      {best_speed['approach']} ({best_speed['fps']:.1f} FPS)\n")
            f.write(f"✓ Best IoU Score:          {best_iou['approach']} ({best_iou['mean_iou']:.3f})\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Summary report saved: {summary_file}")
        
        # Also print to console
        with open(summary_file, 'r') as f:
            print("\n" + f.read())


def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare all approaches')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Test images directory')
    parser.add_argument('--approach1', type=str, 
                       default='results/predictions/approach1_results.json',
                       help='Approach 1 results file')
    parser.add_argument('--approach2', type=str,
                       default='results/predictions/approach2_results.json',
                       help='Approach 2 results file')
    parser.add_argument('--approach3', type=str,
                       default='results/predictions/approach3_results.json',
                       help='Approach 3 results file')
    parser.add_argument('--output_dir', type=str,
                       default='results/comparison',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("EVALUATION AND COMPARISON")
    print("="*70)
    
    # Create evaluator
    evaluator = PerformanceEvaluator(args.test_dir)
    
    # Prepare results files
    results_files = {
        'Approach 1: Traditional CV': args.approach1,
        'Approach 2: YOLO + EasyOCR': args.approach2,
        'Approach 3: Cascade + CNN': args.approach3
    }
    
    # Compare approaches
    comparison_df = evaluator.compare_approaches(results_files)
    
    # Create report
    evaluator.create_comparison_report(comparison_df, args.output_dir)
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
