import pandas as pd
import os
from typing import List, Optional
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

def process_csv_data(
    csv_path: str,
    required_columns: List[str] = [
        "newsID", "transmissionDate", "headline", 
        "story", "isinTags", "tickerTags"
    ],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Process CSV data by filtering out rows with empty headlines and stories,
    and rows where isEnglish is False. Also keep only the specified columns.
    
    Args:
        csv_path: Path to the input CSV file
        required_columns: List of columns to keep in the processed data
        output_path: Optional path to save the processed CSV file
        
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
        
        # Filter rows with non-empty headline and story, and where isEnglish is True
        filtered_df = df[
            df["headline"].notna() & 
            df["story"].notna() & 
            (df["isEnglish"] == True)
        ]
        
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

if __name__ == "__main__":
    # Initialize logging with console output
    setup_logging(log_level="INFO")
    
    # Process the CSV file
    processed_df = process_csv_data(
        csv_path="data/raw/MA2024.csv",
        output_path="data/processed/processed_MA2024.csv"  # Optional
    )
