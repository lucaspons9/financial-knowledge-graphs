from src.utils.reading_files import load_yaml
from src.llm.batch_processor import BatchProcessor


def main():
    # Load sample news data
    try:
        config = load_yaml("configs/config_llm_execution.yaml")
        sample_news = load_yaml(config["data_path"])
        
        # Extract text from sample news
        texts = list(sample_news.values())
        
        print(f"Loaded {len(texts)} sample news texts")
        
        # Initialize BatchProcessor
        batch_processor = BatchProcessor()
        
        # Run batch entity extraction
        print("Running batch entity extraction...")
        entities_batch = batch_processor.run_batch("entity_extraction", texts)
        
        # Print results
        for i, entity_result in enumerate(entities_batch, start=1):
            print(f"Entities in news {i}: {entity_result.content}")
            
        print("\nBatch processing completed successfully!")
        
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 