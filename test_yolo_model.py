"""
TEST TRAINED YOLO MODEL - ENHANCED OCR VERSION
===============================================

This version has MUCH BETTER OCR with:
- Larger image scaling for better text recognition
- Multiple preprocessing strategies
- Tesseract fallback
- Voting system for best result

Usage:
    python test_yolo_enhanced.py --image data/processed/splits/test/images/Cars0.png --visualize
    python test_yolo_enhanced.py --batch data/processed/splits/test/images
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import argparse
from collections import Counter

import torch
import functools

# PyTorch 2.6 fix
_original_torch_load = torch.load
torch.load = functools.partial(_original_torch_load, weights_only=False)

from ultralytics import YOLO
import easyocr


class EnhancedYOLOTester:
    """YOLO tester with ENHANCED OCR capabilities"""

    def __init__(self, model_path):
        print(f"Loading model: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = YOLO(model_path)
        print("✓ YOLO loaded")

        print("Loading EasyOCR...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'), verbose=False)
        print("✓ EasyOCR loaded")
        print(f"Device: {self.device}\n")

    def detect_plate(self, image_path, conf_threshold=0.25):
        """Detect license plate"""
        results = self.model(image_path, conf=conf_threshold, verbose=False)

        if len(results[0].boxes) == 0:
            return None, None, 0.0

        boxes = results[0].boxes
        best_idx = boxes.conf.argmax()
        box = boxes[best_idx]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        image = cv2.imread(str(image_path))
        plate = image[y1:y2, x1:x2]
        bbox = (x1, y1, x2 - x1, y2 - y1)

        return plate, bbox, confidence

    def recognize_text_enhanced(self, plate_image):
        """
        ENHANCED OCR with multiple strategies and voting
        """
        if plate_image is None or plate_image.size == 0:
            return ""

        all_results = []

        # STEP 1: Scale up significantly (CRITICAL!)
        height, width = plate_image.shape[:2]
        if height < 150:
            scale = 250 / height  # Scale to 250px height
            plate_large = cv2.resize(plate_image, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)
        else:
            plate_large = plate_image

        # STRATEGY 1: Original scaled
        try:
            results = self.reader.readtext(plate_large, detail=0, paragraph=False)
            if results:
                text = ''.join(results).upper()
                text = ''.join(c for c in text if c.isalnum())
                if len(text) >= 4:
                    all_results.append(text)
        except:
            pass

        # STRATEGY 2: Grayscale + Strong CLAHE
        try:
            gray = cv2.cvtColor(plate_large, cv2.COLOR_BGR2GRAY) if len(plate_large.shape) == 3 else plate_large
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            results = self.reader.readtext(enhanced_bgr, detail=0, paragraph=False)
            if results:
                text = ''.join(results).upper()
                text = ''.join(c for c in text if c.isalnum())
                if len(text) >= 4:
                    all_results.append(text)
        except:
            pass

        # STRATEGY 3: Binary thresholding
        try:
            gray = cv2.cvtColor(plate_large, cv2.COLOR_BGR2GRAY) if len(plate_large.shape) == 3 else plate_large
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            results = self.reader.readtext(binary_bgr, detail=0, paragraph=False)
            if results:
                text = ''.join(results).upper()
                text = ''.join(c for c in text if c.isalnum())
                if len(text) >= 4:
                    all_results.append(text)
        except:
            pass

        # STRATEGY 4: Inverted binary
        try:
            gray = cv2.cvtColor(plate_large, cv2.COLOR_BGR2GRAY) if len(plate_large.shape) == 3 else plate_large
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

            results = self.reader.readtext(binary_bgr, detail=0, paragraph=False)
            if results:
                text = ''.join(results).upper()
                text = ''.join(c for c in text if c.isalnum())
                if len(text) >= 4:
                    all_results.append(text)
        except:
            pass

        # STRATEGY 5: Tesseract fallback
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

            gray = cv2.cvtColor(plate_large, cv2.COLOR_BGR2GRAY) if len(plate_large.shape) == 3 else plate_large

            # Multiple Tesseract attempts
            configs = [
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]

            for config in configs:
                text = pytesseract.image_to_string(gray, config=config)
                text = text.upper()
                text = ''.join(c for c in text if c.isalnum())
                if len(text) >= 4:
                    all_results.append(text)
        except:
            pass

        if not all_results:
            return ""

        # VOTING: Return most common result
        counter = Counter(all_results)
        if counter:
            most_common = counter.most_common(1)[0][0]
            return most_common

        # Fallback: return longest
        return max(all_results, key=len)

    def process(self, image_path):
        """Full pipeline"""
        plate, bbox, confidence = self.detect_plate(image_path)

        if plate is None:
            return {
                'success': False,
                'text': '',
                'bbox': None,
                'confidence': 0.0
            }

        text = self.recognize_text_enhanced(plate)

        return {
            'success': True if text else False,
            'text': text,
            'bbox': bbox,
            'confidence': confidence,
            'plate_image': plate
        }

    def visualize(self, image_path, result):
        """Visualize results"""
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Detection
        img_viz = image_rgb.copy()
        if result['bbox']:
            x, y, w, h = result['bbox']
            cv2.rectangle(img_viz, (x, y), (x+w, y+h), (0, 255, 0), 3)
            display_text = result['text'] if result['text'] else "NO TEXT DETECTED"
            cv2.putText(img_viz, display_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        axes[0].imshow(img_viz)
        axes[0].set_title(f"YOLO Detection\nText: '{result['text']}'\nConfidence: {result['confidence']:.3f}",
                          fontsize=13, fontweight='bold')
        axes[0].axis('off')

        # Plate
        if 'plate_image' in result and result['plate_image'] is not None:
            plate_rgb = cv2.cvtColor(result['plate_image'], cv2.COLOR_BGR2RGB)
            axes[1].imshow(plate_rgb)
            axes[1].set_title("Extracted Plate", fontsize=13, fontweight='bold')
            axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    def batch_process(self, image_dir, output_file):
        """Process multiple images"""
        image_dir = Path(image_dir)
        images = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

        print(f"\nProcessing {len(images)} images...")
        results = []
        times = []

        for img in tqdm(images):
            start = time.time()
            result = self.process(img)
            elapsed = time.time() - start
            times.append(elapsed)

            results.append({
                'filename': img.name,
                'detected_text': result['text'],
                'confidence': result['confidence'],
                'success': result['success'],
                'processing_time': elapsed
            })

        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Stats
        successful = sum(1 for r in results if r['success'])
        avg_time = np.mean(times)

        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Total:      {len(images)}")
        print(f"Successful: {successful} ({successful/len(images)*100:.1f}%)")
        print(f"Failed:     {len(images)-successful}")
        print(f"Avg time:   {avg_time:.3f}s")
        print(f"FPS:        {1/avg_time:.1f}")
        print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/yolo/license_plate_detector/weights/best.pt')
    parser.add_argument('--image', help='Single image')
    parser.add_argument('--batch', help='Batch directory')
    parser.add_argument('--output', default='results/predictions/approach2_results.json')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    tester = EnhancedYOLOTester(args.model)

    if args.image:
        print(f"\nProcessing: {args.image}")
        result = tester.process(args.image)

        print(f"\n{'='*70}")
        print("RESULT")
        print(f"{'='*70}")
        print(f"Success:    {result['success']}")
        print(f"Text:       '{result['text']}'")
        print(f"Confidence: {result['confidence']:.3f}")

        if args.visualize:
            tester.visualize(args.image, result)

    elif args.batch:
        tester.batch_process(args.batch, args.output)

    else:
        print("Use --image or --batch")


if __name__ == '__main__':
    main()