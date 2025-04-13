import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

def process_csv_data(
    csv_path: str,
    required_columns: List[str] = [
        "newsID", "transmissionDate", "headline", 
        "story", "isinTags", "tickerTags"
    ],
    output_path: Optional[str] = None,
    min_tokens: int = 250
) -> pd.DataFrame:
    """
    Process CSV data by filtering out rows with empty headlines and stories,
    rows where isEnglish is False, and stories with fewer than min_tokens tokens.
    Also keep only the specified columns.
    
    Args:
        csv_path: Path to the input CSV file
        required_columns: List of columns to keep in the processed data
        output_path: Optional path to save the processed CSV file
        min_tokens: Minimum number of tokens required in the story (default: 250)
        
    Returns:
        Processed pandas DataFrame
    """
    logger.info(f"Processing CSV file: {csv_path}")

    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        initial_count = len(df)
        logger.info(f"Loaded CSV with {initial_count} rows")
        
        # Check if all required columns exist in the DataFrame
        missing_columns = [col for col in required_columns + ["isEnglish"] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # Calculate approximate token counts for each story
        df['token_count'] = df['story'].apply(lambda x: len(re.split(r'\s+', str(x))) if pd.notna(x) else 0)
        
        # Filter rows with non-empty headline and story, where isEnglish is True,
        # and story has at least min_tokens tokens
        filtered_df = df[
            df["headline"].notna() & 
            df["story"].notna() & 
            (df["isEnglish"] == True) &
            (df['token_count'] >= min_tokens)
        ]
        
        # Remove duplicate stories
        filtered_df = filtered_df.drop_duplicates(subset='story')
        
        # Keep only the specified columns
        filtered_df = filtered_df[required_columns]
        
        # Log the filtering results
        removed_count = initial_count - len(filtered_df)
        logger.info(f"Removed {removed_count} rows ({removed_count/initial_count:.2%})")
        logger.info(f"Final processed dataframe has {len(filtered_df)} rows")
        
        # Save to file if output_path is provided
        if output_path:
            filtered_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
        
        return filtered_df
    
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        raise

def create_ground_truth_sample(
    input_csv_path: str,
    output_path: str,
    sample_size: int = 100,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create a random sample of news articles for ground truth annotation.
    
    Args:
        input_csv_path: Path to the processed CSV file
        output_path: Path to save the sampled CSV file
        sample_size: Number of news articles to sample (default: 100)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing the sampled news articles
    """
    logger.info(f"Creating ground truth sample from: {input_csv_path}")
    
    # Check if file exists
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"CSV file not found: {input_csv_path}")
    
    try:
        # Read the processed CSV file
        df = pd.read_csv(input_csv_path)
        logger.info(f"Loaded CSV with {len(df)} rows")
        
        # Sample the specified number of news articles
        if len(df) <= sample_size:
            sampled_df = df.copy()
            logger.info(f"Dataset has fewer than {sample_size} rows, using all rows")
        else:
            sampled_df = df.sample(n=sample_size, random_state=random_seed)
            logger.info(f"Sampled {sample_size} rows from the dataset")
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the sampled data to the output path
        sampled_df.to_csv(output_path, index=False)
        logger.info(f"Saved ground truth sample to {output_path}")
        
        return sampled_df
    
    except Exception as e:
        logger.error(f"Error creating ground truth sample: {str(e)}")
        raise

def analyze_token_distribution(
    csv_path: str,
    output_dir: str,
    bin_width: int = 50,
    max_tokens: int = 2000,
    avg_tokens_per_word: float = 1.3
) -> None:
    """
    Analyze and visualize the approximate token distribution of news articles in a CSV file.
    Uses a simple word count approximation instead of a full tokenizer for efficiency.
    
    Args:
        csv_path: Path to the input CSV file containing news articles
        output_dir: Directory to save the generated plots
        bin_width: Width of histogram bins (default: 50)
        max_tokens: Maximum number of tokens to display in the plot (default: 2000)
        avg_tokens_per_word: Average number of tokens per word (default: 1.3)
        
    Returns:
        None
    """
    logger.info(f"Analyzing token distribution in: {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} rows")
    
    # Calculate approximate token counts for each article
    token_counts = []
    for _, row in df.iterrows():
        # Split by whitespace and filter out empty strings
        words = [word for word in re.split(r'\s+', row['story']) if word]

        # Approximate token count (words * avg_tokens_per_word)
        # Add a small buffer for special tokens and punctuation
        approx_tokens = int(len(words) * avg_tokens_per_word) + 10
        token_counts.append(approx_tokens)
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    plt.hist(token_counts, bins=range(200, 2200, bin_width), alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Approximate Distribution of Token Counts in News Articles', fontsize=16)
    plt.xlabel('Approximate Number of Tokens', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add statistics as text
    stats_text = (
        f"Total articles: {len(token_counts)}\n"
        f"Mean tokens: {np.mean(token_counts):.1f}\n"
        f"Median tokens: {np.median(token_counts):.1f}\n"
        f"Min tokens: {min(token_counts)}\n"
        f"Max tokens: {max(token_counts)}\n"
        f"Std dev: {np.std(token_counts):.1f}"
    )
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    output_path = os.path.join(output_dir, "token_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved token distribution plot to {output_path}")
    
    # Print summary statistics
    logger.info(f"Token count statistics:")
    logger.info(f"  Mean: {np.mean(token_counts):.1f}")
    logger.info(f"  Median: {np.median(token_counts):.1f}")
    logger.info(f"  Min: {min(token_counts)}")
    logger.info(f"  Max: {max(token_counts)}")
    logger.info(f"  Std dev: {np.std(token_counts):.1f}")
    
    # Count articles with very few tokens (potential issues)
    short_articles = sum(1 for count in token_counts if count < 50)
    if short_articles > 0:
        logger.warning(f"Found {short_articles} articles with fewer than 50 tokens")
    
    # Find and print examples of shortest and longest articles
    # Create a DataFrame with token counts for easier analysis
    df_with_counts = df.copy()
    df_with_counts['token_count'] = token_counts
    
    # Calculate percentiles
    bottom_10_percentile = np.percentile(token_counts, 10)
    top_10_percentile = np.percentile(token_counts, 90)
    
    # Find examples of shortest and longest articles
    shortest_articles = df_with_counts[df_with_counts['token_count'] <= bottom_10_percentile].head(3)
    longest_articles = df_with_counts[df_with_counts['token_count'] >= top_10_percentile].head(3)
    
    # Print examples
    logger.info("\n--- Examples of shortest articles (bottom 10%) ---")
    for idx, row in shortest_articles.iterrows():
        logger.info(f"\nArticle ID: {row['newsID']}")
        logger.info(f"Token count: {row['token_count']}")
        logger.info(f"Headline: {row['headline']}")
        logger.info(f"Story (first 200 chars): {row['story'][:200]}...")
    
    logger.info("\n--- Examples of longest articles (top 10%) ---")
    for idx, row in longest_articles.iterrows():
        logger.info(f"\nArticle ID: {row['newsID']}")
        logger.info(f"Token count: {row['token_count']}")
        logger.info(f"Headline: {row['headline']}")
        logger.info(f"Story (first 200 chars): {row['story'][:200]}...")

if __name__ == "__main__":
    # Initialize logging with console output
    setup_logging(log_level="INFO")
    
    # # Process the CSV file
    # processed_df = process_csv_data(
    #     csv_path="data/raw/MA2024.csv",
    #     output_path="data/processed/processed_MA2024.csv" 
    # )
    
    # # Create a sample for ground truth annotation
    # ground_truth_sample = create_ground_truth_sample(
    #     input_csv_path="data/processed/processed_MA2024.csv",
    #     output_path="data/processed/ground_truth_sample100.csv"
    # )
    
    # Analyze token distribution in the processed data
    analyze_token_distribution(
        csv_path="data/processed/processed_MA2024.csv",
        output_dir="data/processed/analysis",
    )
