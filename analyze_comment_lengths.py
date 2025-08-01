import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_comment_lengths(csv_file, dataset_name):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    
    # Calculate lengths of each comment
    comment_lengths = []
    for comment in df['Comment']:
        tokens = tokenizer.tokenize(str(comment))
        comment_lengths.append(len(tokens))
    
    # Convert to numpy array for easier calculations
    lengths = np.array(comment_lengths)
    
    # Calculate statistics
    total_comments = len(lengths)
    print(f"\nTotal number of comments: {total_comments}")
    
    # Calculate percentiles
    percentiles = [50, 75, 90, 95, 98]
    print("\nLength percentiles:")
    for p in percentiles:
        length = np.percentile(lengths, p)
        print(f"{p}th percentile: {length:.1f} words")
    
    # Calculate coverage for different length thresholds
    print("\nCoverage analysis:")
    thresholds = [50, 100, 150, 200, 250, 300]
    for threshold in thresholds:
        coverage = (lengths <= threshold).sum() / total_comments * 100
        print(f"Comments with ≤ {threshold} words: {coverage:.1f}%")
    
    # Print some additional statistics
    print(f"\nMinimum length: {lengths.min()}")
    print(f"Maximum length: {lengths.max()}")
    print(f"Mean length: {lengths.mean():.1f}")
    print(f"Median length: {np.median(lengths):.1f}")

    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Histogram of comment lengths
    plt.subplot(2, 1, 1)
    sns.histplot(lengths, bins=50, kde=True)
    plt.title(f'Distribution of Comment Lengths - {dataset_name}')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    
    # Add vertical lines for percentiles
    for p in percentiles:
        length = np.percentile(lengths, p)
        plt.axvline(x=length, color='r', linestyle='--', alpha=0.5)
        plt.text(length, plt.ylim()[1]*0.9, f'{p}th', rotation=90)
    
    # 2. Cumulative distribution
    plt.subplot(2, 1, 2)
    sorted_lengths = np.sort(lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    plt.plot(sorted_lengths, cumulative * 100)
    plt.title(f'Cumulative Distribution of Comment Lengths - {dataset_name}')
    plt.xlabel('Number of Words')
    plt.ylabel('Cumulative Percentage')
    
    # Add horizontal lines for 95% and 98% coverage
    plt.axhline(y=95, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=98, color='g', linestyle='--', alpha=0.5)
    
    # Add vertical lines for thresholds
    for threshold in thresholds:
        coverage = (lengths <= threshold).sum() / total_comments * 100
        plt.axvline(x=threshold, color='gray', linestyle=':', alpha=0.5)
        plt.text(threshold, coverage, f'{coverage:.1f}%', rotation=90)
    
    plt.tight_layout()
    plt.savefig(f'comment_length_analysis_{dataset_name.lower()}.png')
    plt.close()

def main():
    # Analyze both train and test datasets
    print("Analyzing training dataset:")
    analyze_comment_lengths("vispamdetection_dataset/dataset/train.csv", "Training")
    
    print("\nAnalyzing test dataset:")
    analyze_comment_lengths("vispamdetection_dataset/dataset/test.csv", "Test")

if __name__ == "__main__":
    main() 