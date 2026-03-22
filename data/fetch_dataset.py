"""
Data Fetcher - Download Adult Autism Screening Dataset
This script provides multiple options to get the 704-record autism screening dataset
"""

import pandas as pd
import urllib.request
import os

def download_from_url():
    """Download autism screening dataset from public source"""
    print("📥 Downloading Adult Autism Screening Dataset...")
    
    # Option 1: UCI ML Repository (Direct Download)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autism/autism-screening-adult-data.csv"
    
    try:
        print(f"Attempting to download from: {url}")
        df = pd.read_csv(url)
        print(f"✅ Success! Downloaded {len(df)} records with {len(df.columns)} features")
        return df
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        return None

def create_sample_dataset():
    """
    Create a realistic sample dataset based on typical autism screening features
    This is a placeholder if real data can't be downloaded
    """
    print("Creating sample dataset based on AQ-10 screening questionnaire...")
    
    np.random.seed(42)
    n_samples = 704
    
    # Sample features based on common autism screening questionnaires
    data = {
        'A1_social_attention': np.random.randint(0, 2, n_samples),
        'A2_communication': np.random.randint(0, 2, n_samples),
        'A3_focused_attention': np.random.randint(0, 2, n_samples),
        'A4_imagination': np.random.randint(0, 2, n_samples),
        'A5_patterns': np.random.randint(0, 2, n_samples),
        'A6_detailed_memory': np.random.randint(0, 2, n_samples),
        'A7_social_understanding': np.random.randint(0, 2, n_samples),
        'A8_social_relationships': np.random.randint(0, 2, n_samples),
        'A9_anxiety': np.random.randint(0, 2, n_samples),
        'A10_voice_tone_understanding': np.random.randint(0, 2, n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['m', 'f'], n_samples),
        'Class': np.random.choice(['NO', 'YES'], n_samples, p=[0.7, 0.3])  # 70% no autism, 30% yes\n    }\n    \n    df = pd.DataFrame(data)\n    return df

def main():
    print("="*60)
    print("🧠 AUTISM SCREENING DATASET FETCHER")
    print("="*60)
    # Create data directory if needed
    os.makedirs('.', exist_ok=True)
    # Try to download real data
    df = download_from_url()
    if df is None:
        print("\\n⚠️  Could not download from public source.")
        print("Options:")
        print("1. Download manually from Kaggle: https://www.kaggle.com/datasets/fauzanardh/autism-screening-data")
        print("2. Run this script again (check your internet connection)")
        print("3. Place your CSV in this directory as 'autism_screening.csv'")
        return
        # Save to CSV
        output_file = 'autism_screening.csv'
        df.to_csv(output_file, index=False)
        print(f\"\\n✅ Dataset saved to: {output_file}\")
        print(f\"   Shape: {df.shape}\")
        print(f\"   Columns: {list(df.columns)[:10]}...\")
        # Show first 10 columns
    if __name__ == '__main__':
        import numpy as np
        main()