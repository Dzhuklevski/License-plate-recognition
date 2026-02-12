"""
APPROACH 2: YOLOv8 + EasyOCR (ULTRA-ROBUST PYTORCH 2.6+ FIX)
=============================================================

This version completely bypasses the PyTorch 2.6+ weights_only issue
by using the YAML architecture file and letting YOLO handle pretrained
weights internally during training.

Usage:
    python src/approach2_yolo_v2.py --mode train --epochs 50
"""

import cv2
import numpy as np
import os
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import time
import yaml
import warnings

warnings.filterwarnings('ignore')

import torch
import easyocr
from ultralytics import YOLO


class UltraRobustYOLODetector:
    """
    YOLO detector that completely bypasses PyTorch 2.6+ loading issues
    """

    def __init__(self, model_path=None, use_gpu=True):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # For trained models, load directly
        if model_path and Path(model_path).exists():
            print(f"Loading trained model: {model_path}")
            # Trained models don't have the safe_globals issue
            self.model = YOLO(model_path)
        else:
            # For training, use YAML architecture
            # YOLO will handle pretrained weights automatically
            print("Loading YOLOv8n architecture for training")
            self.model = YOLO('yolov8n.yaml')

        # Initialize EasyOCR
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'), verbose=False)
        print("✓ Models loaded successfully!")

    def create_dataset_yaml(self, data_dir, output_path='data/processed/dataset.yaml'):
        """Create dataset configuration"""
        data_dir = Path(data_dir)

        dataset_config = {
            'path': str(data_dir.absolute()),
            'train': 'splits/train/images',
            'val': 'splits/val/images',
            'test': 'splits/test/images',
            'nc': 1,
            'names': ['license_plate']
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"✓ Dataset config created: {output_path}")
        return str(output_path)

    def train(self, data_yaml, epochs=50, img_size=640, batch_size=16):
        """Train YOLO model"""
        print("\n" + "="*70)
        print("TRAINING YOLO MODEL")
        print("="*70)
        print(f"\nThis will take approximately:")
        print(f"  - CPU: 2-3 hours for {epochs} epochs")
        print(f"  - GPU: 20-30 minutes for {epochs} epochs")
        print("\nYOLO will automatically download pretrained weights on first run.\n")

        # Train with pretrained weights (YOLO handles download internally)
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=self.device,
            project='models/yolo',
            name='license_plate_detector',
            pretrained=True,  # Download and use pretrained COCO weights
            patience=10,      # Early stopping
            save=True,
            plots=True,
            verbose=True,
            # Optimization flags
            cache=True,       # Cache images for faster training
            workers=4,        # Number of workers
        )

        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)
        print(f"\nBest model saved to:")
        print(f"  models/yolo/license_plate_detector/weights/best.pt")
        print(f"\nTraining plots saved to:")
        print(f"  models/yolo/license_plate_detector/")

        return results

    def detect_plate(self, image_path, confidence_threshold=0.25):
        """Detect license plate in image"""
        results = self.model(image_path, conf=confidence_threshold, verbose=False)

        if len(results[0].boxes) == 0:
            return None, None, 0.0

        # Get best detection
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax()

        box = boxes[best_idx]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        # Load and extract
        image = cv2.imread(str(image_path))
        plate = image[y1:y2, x1:x2]
        bbox = (x1, y1, x2 - x1, y2 - y1)

        return plate, bbox, confidence

    def preprocess_plate_for_ocr(self, plate_image):
        """Enhanced preprocessing for OCR"""
        if plate_image is None or plate_image.size == 0:
            return plate_image

        # Resize if too small
        height, width = plate_image.shape[:2]
        if height < 80:
            scale = 80 / height
            plate_image = cv2.resize(plate_image, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image

        # Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Convert back to BGR for EasyOCR
        result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

        return result

    def recognize_text(self, plate_image):
        """Recognize text using EasyOCR with multiple strategies"""
        if plate_image is None or plate_image.size == 0:
            return ""

        all_texts = []

        # Strategy 1: Original image
        try:
            results = self.reader.readtext(
                plate_image,
                detail=0,
                paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            if results:
                all_texts.extend(results)
        except:
            pass

        # Strategy 2: Preprocessed image
        try:
            processed = self.preprocess_plate_for_ocr(plate_image)
            results = self.reader.readtext(
                processed,
                detail=0,
                paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            if results:
                all_texts.extend(results)
        except:
            pass

        # Strategy 3: Grayscale only
        try:
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                # Enhance
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                results = self.reader.readtext(
                    gray_bgr,
                    detail=0,
                    paragraph=False,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
                if results:
                    all_texts.extend(results)
        except:
            pass

        if not all_texts:
            return ""

        # Combine and clean
        text = ''.join(all_texts)
        text = text.upper()
        text = ''.join(c for c in text if c.isalnum())

        # Validate
        if len(text) < 4 or len(text) > 12:
            # Try to get most common substring
            if all_texts:
                # Pick longest valid result
                valid = [t for t in all_texts if 4 <= len(t) <= 12]
                if valid:
                    text = max(valid, key=len)
                    text = text.upper()
                    text = ''.join(c for c in text if c.isalnum())

        return text

    def process(self, image_path):
        """Complete pipeline: detect + recognize"""
        plate, bbox, confidence = self.detect_plate(image_path)

        if plate is None:
            return {
                'success': False,
                'text': '',
                'bbox': None,
                'confidence': 0.0,
                'message': 'No license plate detected'
            }

        text = self.recognize_text(plate)

        return {
            'success': True if text else False,
            'text': text,
            'bbox': bbox,
            'confidence': confidence,
            'plate_image': plate
        }

    def visualize_result(self, image_path, result, save_path=None):
        """Visualize detection and recognition"""
        import matplotlib.pyplot as plt

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original with detection
        img_viz = image.copy()
        if result['bbox']:
            x, y, w, h = result['bbox']
            cv2.rectangle(img_viz, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img_viz, result['text'], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        axes[0].imshow(img_viz)
        axes[0].set_title(f"Detection\nText: {result['text']}\nConf: {result['confidence']:.2f}")
        axes[0].axis('off')

        # Extracted plate
        if 'plate_image' in result and result['plate_image'] is not None:
            plate_rgb = cv2.cvtColor(result['plate_image'], cv2.COLOR_BGR2RGB)
            axes[1].imshow(plate_rgb)
            axes[1].set_title("Extracted Plate")
            axes[1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def process_batch(self, image_dir, output_file):
        """Process multiple images"""
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

        print(f"\nProcessing {len(image_files)} images with Approach 2...")

        results = []
        times = []

        for img_path in tqdm(image_files):
            start_time = time.time()
            result = self.process(img_path)
            process_time = time.time() - start_time

            times.append(process_time)

            results.append({
                'filename': img_path.name,
                'detected_text': result['text'],
                'bbox': result['bbox'],
                'confidence': result['confidence'],
                'success': result['success'],
                'processing_time': process_time
            })

        # Save results
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Statistics
        successful = sum(1 for r in results if r['success'])
        avg_time = np.mean(times) if times else 0

        print(f"\n{'='*70}")
        print("APPROACH 2 RESULTS")
        print(f"{'='*70}")
        print(f"Total images:     {len(image_files)}")
        print(f"Successful:       {successful} ({successful/len(image_files)*100:.1f}%)")
        print(f"Failed:           {len(image_files)-successful}")
        print(f"Avg time:         {avg_time:.3f}s")
        print(f"FPS:              {1/avg_time:.1f}" if avg_time > 0 else "FPS: N/A")
        print(f"\nResults saved to: {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Ultra-Robust Approach 2: YOLO + EasyOCR')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'predict', 'test'],
                        help='Mode: train, predict, or test')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (reduce if out of memory)')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--test_dir', type=str, help='Test images directory')
    parser.add_argument('--model', type=str,
                        help='Path to trained model (for predict/test modes)')
    parser.add_argument('--output', type=str,
                        default='results/predictions/approach2_results.json')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization (predict mode)')

    args = parser.parse_args()

    if args.mode == 'train':
        print("\n" + "="*70)
        print("APPROACH 2: TRAINING MODE")
        print("="*70)
        print("\nThis approach uses YOLOv8 for detection + EasyOCR for recognition")
        print("PyTorch 2.6+ compatibility: ✓ Using YAML architecture\n")

        detector = UltraRobustYOLODetector()
        dataset_yaml = detector.create_dataset_yaml(args.data_dir)
        detector.train(dataset_yaml, epochs=args.epochs, batch_size=args.batch_size)

        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Test on single image:")
        print("   python src/approach2_yolo_v2.py --mode predict \\")
        print("          --image data/processed/splits/test/images/Cars0.png \\")
        print("          --model models/yolo/license_plate_detector/weights/best.pt \\")
        print("          --visualize")
        print("\n2. Test on all test images:")
        print("   python src/approach2_yolo_v2.py --mode test \\")
        print("          --test_dir data/processed/splits/test/images \\")
        print("          --model models/yolo/license_plate_detector/weights/best.pt")

    elif args.mode == 'predict':
        if not args.image:
            print("Error: --image required for predict mode")
            return

        if not args.model:
            print("Error: --model required for predict mode")
            print("Use: --model models/yolo/license_plate_detector/weights/best.pt")
            return

        detector = UltraRobustYOLODetector(model_path=args.model)

        print(f"\nProcessing: {args.image}")
        result = detector.process(args.image)

        print(f"\n{'='*70}")
        print("RESULT")
        print(f"{'='*70}")
        print(f"Success:    {result['success']}")
        print(f"Text:       {result['text']}")
        print(f"Confidence: {result['confidence']:.3f}")

        if args.visualize:
            detector.visualize_result(args.image, result)

    elif args.mode == 'test':
        if not args.test_dir:
            print("Error: --test_dir required for test mode")
            return

        if not args.model:
            print("Error: --model required for test mode")
            print("Use: --model models/yolo/license_plate_detector/weights/best.pt")
            return

        detector = UltraRobustYOLODetector(model_path=args.model)
        detector.process_batch(args.test_dir, args.output)

        print("\n" + "="*70)
        print("NEXT STEP: CHECK ACCURACY")
        print("="*70)
        print(f"\npython check_accuracy.py {args.output}")


if __name__ == '__main__':
    main()