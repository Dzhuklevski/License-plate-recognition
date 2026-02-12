"""
STEP 4: Data Preprocessing Script
==================================
This script:
1. Loads annotations from XML files
2. Analyzes the dataset
3. Creates train/val/test splits
4. Converts to YOLO format for Approach 2
5. Saves everything in organized folders

Usage:
    python src/preprocessing.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


class DatasetPreprocessor:
    def __init__(self, raw_data_dir, output_dir):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.annotations_data = []

    def parse_xml_annotation(self, xml_path):
        """Extract bounding box from XML annotation"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find('filename').text

        # Get image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Get bounding box
        obj = root.find('object')
        bbox = obj.find('bndbox')

        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        return {
            'filename': filename,
            'width': width,
            'height': height,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        }

    def load_dataset(self):
        """Load all annotations"""
        print("\n" + "="*70)
        print("LOADING DATASET")
        print("="*70)

        annotations_dir = self.raw_data_dir / 'annotations'
        xml_files = list(annotations_dir.glob('*.xml'))

        print(f"Found {len(xml_files)} annotation files")

        for xml_file in tqdm(xml_files, desc="Parsing annotations"):
            try:
                annotation = self.parse_xml_annotation(xml_file)
                self.annotations_data.append(annotation)
            except Exception as e:
                print(f"Error parsing {xml_file.name}: {e}")

        self.annotations_df = pd.DataFrame(self.annotations_data)
        print(f"Successfully loaded {len(self.annotations_df)} annotations")

        return self.annotations_df

    def analyze_dataset(self):
        """Generate statistics"""
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)

        df = self.annotations_df

        print(f"\nTotal images: {len(df)}")
        print(f"\nImage dimensions:")
        print(f"  Width  - Mean: {df['width'].mean():.0f}, Min: {df['width'].min()}, Max: {df['width'].max()}")
        print(f"  Height - Mean: {df['height'].mean():.0f}, Min: {df['height'].min()}, Max: {df['height'].max()}")

        df['bbox_width'] = df['xmax'] - df['xmin']
        df['bbox_height'] = df['ymax'] - df['ymin']

        print(f"\nBounding box dimensions:")
        print(f"  Width  - Mean: {df['bbox_width'].mean():.0f}, Min: {df['bbox_width'].min()}, Max: {df['bbox_width'].max()}")
        print(f"  Height - Mean: {df['bbox_height'].mean():.0f}, Min: {df['bbox_height'].min()}, Max: {df['bbox_height'].max()}")

        df['aspect_ratio'] = df['bbox_width'] / df['bbox_height']
        print(f"\nAspect ratio: Mean: {df['aspect_ratio'].mean():.2f}")

        # Save statistics
        with open(self.output_dir / 'dataset_statistics.txt', 'w') as f:
            f.write(df.describe().to_string())

        print(f"\nStatistics saved to: {self.output_dir / 'dataset_statistics.txt'}")

    def create_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train/val/test"""
        print("\n" + "="*70)
        print(f"CREATING SPLITS (Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio})")
        print("="*70)

        # Split
        train_df, temp_df = train_test_split(
            self.annotations_df,
            test_size=(val_ratio + test_ratio),
            random_state=42
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=42
        )

        print(f"\nTrain: {len(train_df)} images")
        print(f"Val:   {len(val_df)} images")
        print(f"Test:  {len(test_df)} images")

        # Create directories
        splits_dir = self.output_dir / 'splits'
        for split in ['train', 'val', 'test']:
            (splits_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (splits_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Copy files and create YOLO labels
        self.copy_and_convert(train_df, 'train', splits_dir)
        self.copy_and_convert(val_df, 'val', splits_dir)
        self.copy_and_convert(test_df, 'test', splits_dir)

        # Save split info
        split_info = {
            'train': train_df['filename'].tolist(),
            'val': val_df['filename'].tolist(),
            'test': test_df['filename'].tolist()
        }

        with open(splits_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)

        print("\n✓ Dataset splits created successfully!")

    def copy_and_convert(self, split_df, split_name, splits_dir):
        """Copy images and create YOLO format labels"""
        images_src = self.raw_data_dir / 'images'
        images_dst = splits_dir / split_name / 'images'
        labels_dst = splits_dir / split_name / 'labels'

        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
            filename = row['filename']

            # Copy image
            src_path = images_src / filename
            if src_path.exists():
                shutil.copy2(src_path, images_dst / filename)

                # Create YOLO label
                img_width = row['width']
                img_height = row['height']

                # Convert to YOLO format (normalized center_x, center_y, width, height)
                center_x = ((row['xmin'] + row['xmax']) / 2) / img_width
                center_y = ((row['ymin'] + row['ymax']) / 2) / img_height
                bbox_width = (row['xmax'] - row['xmin']) / img_width
                bbox_height = (row['ymax'] - row['ymin']) / img_height

                # Save label file
                label_filename = filename.replace('.png', '.txt').replace('.jpg', '.txt')
                with open(labels_dst / label_filename, 'w') as f:
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    def visualize_samples(self, num_samples=6):
        """Visualize random samples"""
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)

        samples = self.annotations_df.sample(n=min(num_samples, len(self.annotations_df)))

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        images_dir = self.raw_data_dir / 'images'

        for idx, (_, row) in enumerate(samples.iterrows()):
            if idx >= num_samples:
                break

            img_path = images_dir / row['filename']
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Draw bounding box
            cv2.rectangle(img,
                          (row['xmin'], row['ymin']),
                          (row['xmax'], row['ymax']),
                          (0, 255, 0), 3)

            axes[idx].imshow(img)
            axes[idx].set_title(f"{row['filename']}")
            axes[idx].axis('off')

        plt.tight_layout()
        viz_path = self.output_dir / 'sample_images.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualizations saved to: {viz_path}")
        plt.close()


def main():
    # Configuration
    BASE_DIR = Path(r"C:\Users\dzukl\OneDrive\Desktop\DPNS\license-plate-detection")
    RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "car-plate-detection"
    OUTPUT_DIR = BASE_DIR / "data" / "processed"

    print("="*70)
    print("LICENSE PLATE DATASET PREPROCESSING")
    print("="*70)
    print(f"\nRaw data: {RAW_DATA_DIR}")
    print(f"Output:   {OUTPUT_DIR}")

    # Create preprocessor
    preprocessor = DatasetPreprocessor(RAW_DATA_DIR, OUTPUT_DIR)

    # Execute pipeline
    preprocessor.load_dataset()
    preprocessor.analyze_dataset()
    preprocessor.visualize_samples()
    preprocessor.create_splits()

    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nProcessed data location: {OUTPUT_DIR}")
    print("\nNext step: Run Approach 1 (Traditional CV)")


if __name__ == '__main__':
    main()