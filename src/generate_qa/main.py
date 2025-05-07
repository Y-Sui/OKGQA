import os
import pandas as pd
import ast
from datetime import datetime
from .generate_query import multi_process_query
from .post_process import post_process, verify_and_filter_entities, retrieve_wikipedia_pages
from .calculate_stat import calculate_stats, plot_stats, plot_pan_stats
from ..config.generate_qa_config import (
    SEED_SAMPLE_SIZE, RE_SAMPLE, TIMESTAMP_FORMAT,      
    QUERY_DIR, PLOTS_DIR, WIKI_DIR
)

def main():
    """
    Main function to orchestrate the QA generation pipeline.
    
    Workflow:
    1. Generates raw queries using LLM
    2. Post-processes queries to remove duplicates and validate format
    3. Verifies and filters entities by checking their URLs
    4. Retrieves Wikipedia pages for valid entities
    5. Calculates and plots statistics about the generated dataset
    
    The function handles both new generation and loading of existing datasets
    based on the RE_SAMPLE configuration.
    """
    # Configuration
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    os.makedirs(QUERY_DIR, exist_ok=True)
    os.makedirs(WIKI_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Generate raw queries
    raw_dataset_path = os.path.join(QUERY_DIR, f"questions_{timestamp}_{SEED_SAMPLE_SIZE}.csv")
    
    print(f"Generating queries based on {SEED_SAMPLE_SIZE} seed instructions...")
    if not os.path.exists(raw_dataset_path) or RE_SAMPLE:
        df_raw = multi_process_query(raw_dataset_path, seed_sample_size=SEED_SAMPLE_SIZE)
    else:
        df_raw = pd.read_csv(raw_dataset_path)
        try:
            df_raw['dbpedia_entities'] = df_raw['dbpedia_entities'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            df_raw['placeholders'] = df_raw['placeholders'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing data: {e}")
            print("Please check the format of dbpedia_entities and placeholders columns")
            raise
    
    # Post-process the generated queries
    print("\nPost-processing queries...")
    df_post_processed = post_process(df_raw)
    
    # Verify and filter entities
    print("\nVerifying and filtering entities...")
    df_final = verify_and_filter_entities(df_post_processed)
    
    # Save final results
    print(f"\nTotal queries generated: {len(df_raw)}")      
    print(f"Queries after post-processing: {len(df_post_processed)}")
    print(f"Final valid queries: {len(df_final)}")
    
    # Retrieve wikipedia pages
    print("\nRetrieving wikipedia pages...")
    retrieve_wikipedia_pages(df_final, WIKI_DIR)
    df_final.to_csv(os.path.join(QUERY_DIR, f"questions_{timestamp}_{SEED_SAMPLE_SIZE}_post_processed.csv"))
    
    # Calculate statistics
    type_counts, type_naturalness_counts, type_difficulty_counts = calculate_stats(df_final)

    # Plot statistics
    plot_stats(type_counts, type_naturalness_counts, type_difficulty_counts, PLOTS_DIR)
    plot_pan_stats(type_counts, PLOTS_DIR)

if __name__ == "__main__":
    main() 