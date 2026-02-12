"""
APPROACH 1: Traditional Computer Vision + Tesseract OCR (FIXED VERSION)
========================================================================

IMPROVEMENTS:
1. Better image preprocessing for OCR
2. Multiple preprocessing strategies
3. Improved text cleaning and validation
4. Better contour filtering
5. Enhanced OCR configuration

Usage:
    python src/approach1_traditional_fixed.py --batch data/processed/splits/test/images --output results/predictions/approach1_fixed_results.json
"""

import cv2
import numpy as np
import pytesseract
import re
import os
from pathlib import Path
import json
import argparse
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# IMPORTANT: Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class ImprovedLicensePlateDetector:
    """
    Improved license plate detector with better OCR
    """

    def __init__(self,
                 min_area=500,
                 max_area=50000,
                 min_aspect_ratio=2.0,
                 max_aspect_ratio=6.0,
                 debug=False):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.debug = debug

    def preprocess_image(self, image):
        """Enhanced preprocessing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        return gray

    def detect_edges(self, gray_image):
        """Improved edge detection"""
        # Multiple edge detection strategies
        edges1 = cv2.Canny(gray_image, 30, 200)
        edges2 = cv2.Canny(gray_image, 50, 150)

        # Combine edges
        edges = cv2.bitwise_or(edges1, edges2)

        # Morphological closing to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return edges

    def find_license_plate_contours(self, edges, original_shape):
        """Find potential license plate contours"""
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        image_area = original_shape[0] * original_shape[1]
        candidates = []

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Filter by aspect ratio
            aspect_ratio = w / float(h)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            # Additional filters
            if h < 15 or w < 40:  # Too small
                continue

            # Calculate score
            score = self.calculate_score(contour, approx, area, aspect_ratio, image_area)

            candidates.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'score': score,
                'vertices': len(approx)
            })

        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates

    def calculate_score(self, contour, approx, area, aspect_ratio, image_area):
        """Calculate confidence score"""
        score = 0

        # Rectangular shape (4 vertices is ideal)
        if len(approx) == 4:
            score += 40
        elif len(approx) >= 4 and len(approx) <= 8:
            score += 25

        # Aspect ratio (3.0-4.5 is typical for plates)
        ideal_ratio = 3.5
        ratio_diff = abs(aspect_ratio - ideal_ratio)
        ratio_score = max(0, 30 - ratio_diff * 8)
        score += ratio_score

        # Relative size (0.5% - 10% of image)
        relative_size = area / image_area
        if 0.005 <= relative_size <= 0.10:
            score += 20
        elif 0.003 <= relative_size <= 0.15:
            score += 10

        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            score += solidity * 10

        return score

    def extract_plate_region(self, image, bbox):
        """Extract license plate with padding"""
        x, y, w, h = bbox

        # Add padding
        padding_x = int(w * 0.05)
        padding_y = int(h * 0.1)

        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(image.shape[1] - x, w + 2 * padding_x)
        h = min(image.shape[0] - y, h + 2 * padding_y)

        plate = image[y:y+h, x:x+w]
        return plate

    def preprocess_for_ocr_v1(self, plate_image):
        """OCR Preprocessing Strategy 1: Adaptive Threshold"""
        # Resize to better resolution
        height = plate_image.shape[0]
        if height < 80:
            scale = 80 / height
            plate_image = cv2.resize(plate_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()

        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 10
        )

        # Morphological operations
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def preprocess_for_ocr_v2(self, plate_image):
        """OCR Preprocessing Strategy 2: Otsu's Threshold"""
        height = plate_image.shape[0]
        if height < 80:
            scale = 80 / height
            plate_image = cv2.resize(plate_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()

        # Enhance contrast
        gray = cv2.equalizeHist(gray)

        # Blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu's threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def preprocess_for_ocr_v3(self, plate_image):
        """OCR Preprocessing Strategy 3: Enhanced with dilation"""
        height = plate_image.shape[0]
        if height < 80:
            scale = 80 / height
            plate_image = cv2.resize(plate_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        # Bilateral filter
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dilate to make characters thicker
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

        return binary

    def clean_text(self, text):
        """Clean and validate OCR text"""
        if not text:
            return ""

        # Remove all non-alphanumeric characters
        text = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Common OCR mistakes
        replacements = {
            'O': '0',  # O -> 0 (when surrounded by numbers)
            'I': '1',  # I -> 1 (when surrounded by numbers)
            'Z': '2',  # Z -> 2 (sometimes)
            'S': '5',  # S -> 5 (sometimes)
            'B': '8',  # B -> 8 (sometimes)
        }

        # Only apply replacements if text looks like it has numbers
        if any(c.isdigit() for c in text):
            # Keep the text as-is mostly, just remove clearly wrong chars
            pass

        return text

    def is_valid_plate_text(self, text):
        """Validate if text looks like a license plate"""
        if not text:
            return False

        # Minimum length
        if len(text) < 4:
            return False

        # Maximum length
        if len(text) > 10:
            return False

        # Should have mix of letters and/or numbers
        has_letter = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)

        # At least one of each is good, or all letters/all digits for some plates
        return True  # Be lenient for now

    def recognize_text(self, plate_image):
        """Try multiple OCR strategies and pick best result"""
        # Tesseract configurations
        configs = [
            '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        ]

        all_results = []

        # Try different preprocessing methods
        preprocessed_images = [
            self.preprocess_for_ocr_v1(plate_image),
            self.preprocess_for_ocr_v2(plate_image),
            self.preprocess_for_ocr_v3(plate_image),
            cv2.bitwise_not(self.preprocess_for_ocr_v1(plate_image)),  # Inverted v1
            cv2.bitwise_not(self.preprocess_for_ocr_v2(plate_image)),  # Inverted v2
        ]

        # Try each combination
        for img_processed in preprocessed_images:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(img_processed, config=config)
                    text = self.clean_text(text)
                    if self.is_valid_plate_text(text):
                        all_results.append(text)
                except Exception as e:
                    continue

        # If no valid results, return empty
        if not all_results:
            return ""

        # Pick the most common result (voting)
        from collections import Counter
        text_counts = Counter(all_results)
        most_common = text_counts.most_common(1)[0][0]

        # If there's a tie, pick the longest
        max_count = text_counts.most_common(1)[0][1]
        top_results = [text for text, count in text_counts.items() if count == max_count]

        if len(top_results) > 1:
            return max(top_results, key=len)

        return most_common

    def detect_and_recognize(self, image_path):
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

        # Pipeline
        gray = self.preprocess_image(image)
        edges = self.detect_edges(gray)
        candidates = self.find_license_plate_contours(edges, image.shape)

        if not candidates:
            return {
                'success': False,
                'text': '',
                'bbox': None,
                'confidence': 0,
                'message': 'No license plate detected'
            }

        # Try top 3 candidates
        for candidate in candidates[:3]:
            bbox = candidate['bbox']
            plate_region = self.extract_plate_region(image, bbox)
            text = self.recognize_text(plate_region)

            if text and self.is_valid_plate_text(text):
                # Store processed plate for visualization
                processed_plate = self.preprocess_for_ocr_v1(plate_region)
                return {
                    'success': True,
                    'text': text,
                    'bbox': bbox,
                    'confidence': candidate['score'],
                    'plate_image': plate_region,
                    'processed_plate': processed_plate
                }

        # If nothing valid, return best candidate anyway
        best = candidates[0]
        bbox = best['bbox']
        plate_region = self.extract_plate_region(image, bbox)
        text = self.recognize_text(plate_region)
        processed_plate = self.preprocess_for_ocr_v1(plate_region)

        return {
            'success': True if text else False,
            'text': text,
            'bbox': bbox,
            'confidence': best['score'],
            'plate_image': plate_region,
            'processed_plate': processed_plate
        }

    def visualize_result(self, image_path, result, save_path=None):
        """Visualize detection result"""
        image = cv2.imread(str(image_path))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original with detection
        img_viz = image.copy()
        if result['bbox']:
            x, y, w, h = result['bbox']
            cv2.rectangle(img_viz, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img_viz, result['text'], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        axes[0].imshow(cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Detection\nText: {result['text']}\nScore: {result['confidence']:.1f}")
        axes[0].axis('off')

        # Extracted plate
        if 'plate_image' in result and result['plate_image'] is not None:
            axes[1].imshow(cv2.cvtColor(result['plate_image'], cv2.COLOR_BGR2RGB))
            axes[1].set_title("Extracted Plate")
            axes[1].axis('off')

        # Processed plate
        if 'processed_plate' in result and result['processed_plate'] is not None:
            axes[2].imshow(result['processed_plate'], cmap='gray')
            axes[2].set_title("Processed for OCR")
            axes[2].axis('off')

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

        print(f"\nProcessing {len(image_files)} images with Improved Approach 1...")

        results = []
        times = []

        for img_path in tqdm(image_files):
            start_time = time.time()
            result = self.detect_and_recognize(img_path)
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
        avg_time = np.mean(times)

        print(f"\n{'='*70}")
        print("IMPROVED APPROACH 1 RESULTS")
        print(f"{'='*70}")
        print(f"Total images:     {len(image_files)}")
        print(f"Successful:       {successful} ({successful/len(image_files)*100:.1f}%)")
        print(f"Failed:           {len(image_files)-successful}")
        print(f"Avg time:         {avg_time:.3f}s")
        print(f"FPS:              {1/avg_time:.1f}")
        print(f"\nResults saved to: {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Improved Approach 1')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--batch', type=str, help='Directory of images')
    parser.add_argument('--output', type=str, default='results/predictions/approach1_fixed_results.json')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    detector = ImprovedLicensePlateDetector(debug=args.debug)

    if args.image:
        print(f"\nProcessing: {args.image}")
        result = detector.detect_and_recognize(args.image)

        print(f"\n{'='*70}")
        print("RESULT")
        print(f"{'='*70}")
        print(f"Success:    {result['success']}")
        print(f"Text:       {result['text']}")
        print(f"Confidence: {result['confidence']:.2f}")

        if args.visualize:
            detector.visualize_result(args.image, result)

    elif args.batch:
        detector.process_batch(args.batch, args.output)

    else:
        print("Please provide --image or --batch argument")


if __name__ == '__main__':
    main()