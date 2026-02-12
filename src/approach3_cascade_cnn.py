"""
APPROACH 3: Smart Adaptive Haar Cascade + OCR
===============================================

This version automatically detects:
- Indian plates with blue IND badge → removes badge
- Standard white plates → keeps full plate
- Adapts preprocessing based on plate characteristics

Usage:
    python approach3_smart.py --batch data/processed/splits/test/images --output results/predictions/approach3_results.json
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import time
import re
from collections import Counter
import matplotlib.pyplot as plt

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    print("Warning: pytesseract not installed")


class SmartAdaptiveDetector:
    """
    Smart detector that adapts to different plate types
    """

    def __init__(self, cascade_path=None, debug=False):
        self.debug = debug

        # Load Haar Cascade
        if cascade_path and Path(cascade_path).exists():
            self.cascade = cv2.CascadeClassifier(cascade_path)
        else:
            cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            self.cascade = cv2.CascadeClassifier(cascade_path)

        if self.cascade.empty():
            raise ValueError("Failed to load Haar Cascade!")

        print("✓ Smart Adaptive Detector initialized")

    def detect_plate_cascade(self, image):
        """Detect license plate using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        plates = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(60, 20),
            maxSize=(400, 150)
        )

        if len(plates) == 0:
            return None, None

        # Take largest detection
        areas = [w * h for (x, y, w, h) in plates]
        best_idx = np.argmax(areas)
        x, y, w, h = plates[best_idx]

        # Extract with padding
        padding_x = int(w * 0.05)
        padding_y = int(h * 0.1)

        x_start = max(0, x - padding_x)
        y_start = max(0, y - padding_y)
        x_end = min(image.shape[1], x + w + padding_x)
        y_end = min(image.shape[0], y + h + padding_y)

        plate = image[y_start:y_end, x_start:x_end]
        bbox = (x, y, w, h)

        return plate, bbox

    def has_blue_badge(self, plate_image):
        """
        Detect if plate has blue IND badge on left side

        Returns:
            bool: True if blue badge detected, False otherwise
        """
        if plate_image is None or plate_image.size == 0:
            return False

        height, width = plate_image.shape[:2]

        # Check left 25% of plate for blue color
        left_region = plate_image[:, :int(width * 0.25)]

        # Convert to HSV for blue detection
        if len(left_region.shape) == 3:
            hsv = cv2.cvtColor(left_region, cv2.COLOR_BGR2HSV)
        else:
            return False  # Grayscale, no color info

        # Blue color range in HSV
        # Hue: 100-130, Saturation: 50-255, Value: 50-255
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Create mask for blue color
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Calculate percentage of blue pixels
        blue_pixels = np.sum(blue_mask > 0)
        total_pixels = blue_mask.size
        blue_percentage = (blue_pixels / total_pixels) * 100

        # If more than 15% of left region is blue, likely has badge
        has_badge = blue_percentage > 15

        if self.debug:
            print(f"  Blue detection: {blue_percentage:.1f}% → Badge: {has_badge}")

        return has_badge

    def smart_crop_plate(self, plate_image):
        """
        Intelligently crop plate based on whether it has a badge

        Returns:
            tuple: (cropped_image, has_badge)
        """
        has_badge = self.has_blue_badge(plate_image)

        if has_badge:
            # Remove left portion (blue badge)
            width = plate_image.shape[1]
            crop_amount = int(width * 0.18)  # Remove left 18%
            cropped = plate_image[:, crop_amount:]

            if self.debug:
                print("  → Cropping badge (Indian plate)")
        else:
            # Keep full plate (no badge)
            cropped = plate_image.copy()

            if self.debug:
                print("  → Keeping full plate (standard plate)")

        return cropped, has_badge

    def preprocess_aggressive(self, plate_image, has_badge):
        """
        Aggressive preprocessing - adapts based on badge presence
        """
        # Smart crop first
        plate, _ = self.smart_crop_plate(plate_image)

        if plate is None or plate.size == 0:
            return plate_image

        # Resize to consistent height
        target_height = 120
        aspect = plate.shape[1] / plate.shape[0]
        target_width = int(target_height * aspect)
        plate = cv2.resize(plate, (target_width, target_height),
                           interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        if len(plate.shape) == 3:
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate

        # Remove shadows (more aggressive if badge detected)
        kernel_size = 15 if has_badge else 11
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.divide(gray, background, scale=255)

        # Denoise
        denoise_h = 15 if has_badge else 10
        gray = cv2.fastNlMeansDenoising(gray, h=denoise_h)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        # Sharpen
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel_sharpen)

        # Threshold
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dilate slightly
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        return binary

    def preprocess_standard(self, plate_image, has_badge):
        """
        Standard preprocessing
        """
        plate, _ = self.smart_crop_plate(plate_image)

        if plate is None or plate.size == 0:
            return plate_image

        # Resize
        height = plate.shape[0]
        if height < 100:
            scale = 100 / height
            plate = cv2.resize(plate, None, fx=scale, fy=scale)

        # Grayscale
        if len(plate.shape) == 3:
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate

        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 10
        )

        return binary

    def preprocess_otsu(self, plate_image, has_badge):
        """
        Otsu thresholding
        """
        plate, _ = self.smart_crop_plate(plate_image)

        if plate is None or plate.size == 0:
            return plate_image

        # Resize
        height = plate.shape[0]
        if height < 100:
            scale = 100 / height
            plate = cv2.resize(plate, None, fx=scale, fy=scale)

        # Grayscale
        if len(plate.shape) == 3:
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate

        # Histogram equalization
        gray = cv2.equalizeHist(gray)

        # Gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu threshold
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def clean_text(self, text):
        """Clean OCR text"""
        if not text:
            return ""

        # Remove non-alphanumeric
        text = re.sub(r'[^A-Z0-9]', '', text.upper())

        return text

    def is_valid_plate_text(self, text):
        """Validate plate text"""
        if not text:
            return False

        # Length check
        if len(text) < 4 or len(text) > 15:
            return False

        # Must have both letters and numbers
        has_letter = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)

        if not (has_letter and has_digit):
            return False

        return True

    def recognize_text(self, plate_image):
        """
        Smart OCR with adaptive preprocessing
        """
        # Detect badge presence ONCE
        has_badge = self.has_blue_badge(plate_image)

        if self.debug:
            print(f"\nPlate type: {'Indian (with badge)' if has_badge else 'Standard (no badge)'}")

        # Tesseract configs
        configs = [
            '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]

        all_results = []

        # Preprocessing strategies (adapted to badge presence)
        preprocessed_images = [
            self.preprocess_aggressive(plate_image, has_badge),
            self.preprocess_standard(plate_image, has_badge),
            self.preprocess_otsu(plate_image, has_badge),
            cv2.bitwise_not(self.preprocess_aggressive(plate_image, has_badge)),
            cv2.bitwise_not(self.preprocess_standard(plate_image, has_badge)),
        ]

        # Try all combinations
        for img_processed in preprocessed_images:
            if img_processed is None or img_processed.size == 0:
                continue

            for config in configs:
                try:
                    text = pytesseract.image_to_string(img_processed, config=config)
                    text = self.clean_text(text)

                    if self.is_valid_plate_text(text):
                        all_results.append(text)

                except Exception as e:
                    if self.debug:
                        print(f"  OCR error: {e}")
                    continue

        if not all_results:
            return ""

        # Voting
        text_counts = Counter(all_results)
        top_results = text_counts.most_common(3)

        # If clear winner (3+ occurrences)
        if top_results[0][1] >= 3:
            return top_results[0][0]

        # Pick longest valid from top results
        top_texts = [text for text, count in top_results]
        valid_texts = [t for t in top_texts if self.is_valid_plate_text(t)]

        if valid_texts:
            return max(valid_texts, key=len)

        return top_results[0][0]

    def process(self, image_path):
        """Complete pipeline"""
        image = cv2.imread(str(image_path))
        if image is None:
            return {
                'success': False,
                'text': '',
                'bbox': None,
                'confidence': 0,
                'error': 'Could not load image'
            }

        plate, bbox = self.detect_plate_cascade(image)

        if plate is None:
            return {
                'success': False,
                'text': '',
                'bbox': None,
                'confidence': 0,
                'message': 'No license plate detected'
            }

        text = self.recognize_text(plate)
        confidence = min(100, len(text) * 10) if text else 0

        return {
            'success': True if text else False,
            'text': text,
            'bbox': bbox,
            'confidence': confidence,
            'plate_image': plate
        }

    def visualize_result(self, image_path, result, save_path=None):
        """Visualize detection result"""
        image = cv2.imread(str(image_path))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Original with detection
        img_viz = image.copy()
        if result['bbox']:
            x, y, w, h = result['bbox']
            cv2.rectangle(img_viz, (x, y), (x+w, y+h), (0, 255, 0), 3)
            text_display = result['text'] if result['text'] else "NO TEXT"
            cv2.putText(img_viz, text_display, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        axes[0].imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Detection\nText: '{result['text']}'\nConf: {result['confidence']:.1f}",
                          fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Extracted plate
        if 'plate_image' in result and result['plate_image'] is not None:
            plate_rgb = cv2.cvtColor(result['plate_image'], cv2.COLOR_BGR2RGB)
            axes[1].imshow(plate_rgb)
            axes[1].set_title("Extracted Plate", fontsize=12, fontweight='bold')
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

        if not image_files:
            print(f"No images found in {image_dir}")
            return

        print(f"\nProcessing {len(image_files)} images with Smart Adaptive Approach 3...")

        results = []
        times = []

        for img_path in tqdm(image_files, desc="Testing"):
            start_time = time.time()
            result = self.process(img_path)
            process_time = time.time() - start_time

            times.append(process_time)

            results.append({
                'filename': img_path.name,
                'detected_text': result['text'],
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
        print("SMART ADAPTIVE APPROACH 3 RESULTS")
        print(f"{'='*70}")
        print(f"Total images:     {len(image_files)}")
        print(f"Successful:       {successful} ({successful/len(image_files)*100:.1f}%)")
        print(f"Failed:           {len(image_files)-successful}")
        print(f"Avg time:         {avg_time:.3f}s")
        print(f"FPS:              {1/avg_time:.1f}" if avg_time > 0 else "FPS: N/A")
        print(f"\nResults saved to: {output_path}")
        print(f"\nNext step:")
        print(f"  python check_accuracy.py {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Smart Adaptive Approach 3')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--batch', type=str, help='Batch directory')
    parser.add_argument('--cascade', type=str, help='Cascade path (optional)')
    parser.add_argument('--output', type=str,
                        default='results/predictions/approach3_results.json')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    try:
        detector = SmartAdaptiveDetector(
            cascade_path=args.cascade,
            debug=args.debug
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return

    if args.image:
        print(f"\n{'='*70}")
        print("TESTING SINGLE IMAGE")
        print(f"{'='*70}\n")

        result = detector.process(args.image)

        print(f"{'='*70}")
        print("RESULT")
        print(f"{'='*70}")
        print(f"Success:         {result['success']}")
        print(f"Detected Text:   '{result['text']}'")
        print(f"Confidence:      {result['confidence']:.1f}")

        if args.visualize:
            detector.visualize_result(args.image, result)

    elif args.batch:
        detector.process_batch(args.batch, args.output)

    else:
        print("\n❌ Error: Provide --image or --batch")
        print("\nExamples:")
        print("  python approach3_smart.py --image test.png --visualize")
        print("  python approach3_smart.py --batch data/processed/splits/test/images")


if __name__ == '__main__':
    main()