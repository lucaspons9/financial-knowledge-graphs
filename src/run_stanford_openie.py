"""
Stanford OpenIE Extraction Runner

This script runs the Stanford OpenIE extraction on sample news data.
"""

from src.utils.reading_files import load_yaml
from src.utils.ground_truth import StanfordOpenIEExtractor


def main():
    # Load sample news data
    try:
        config = load_yaml("configs/config_stanford_openie.yaml")
        data_path = config.get("data_path", "data/raw/sample_news.yaml")
        sample_news = load_yaml(data_path)
        
        # Extract text from sample news
        texts = list(sample_news.values())
        print(f"Loaded {len(texts)} sample news texts")
        
        # Initialize Stanford OpenIE extractor
        print("Initializing Stanford OpenIE extractor...")
        openie_extractor = StanfordOpenIEExtractor(config_path="configs/config_stanford_openie.yaml")
        
        # Extract ground truth triples
        print("Extracting ground truth triples...")
        batch_name = config.get("batch_name", "news")
        triples_batch = openie_extractor.batch_extract(texts, batch_name)
        
        # Print summary of extracted triples
        total_triples = sum(len(triples) for triples in triples_batch)
        print(f"Extracted {total_triples} triples as ground truth")
        print(f"Ground truth data saved to {openie_extractor.output_dir}")
            
        print("\nGround truth extraction completed successfully!")
        
    except Exception as e:
        print(f"Error during ground truth extraction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 