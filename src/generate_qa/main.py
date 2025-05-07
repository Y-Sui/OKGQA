import os
import pandas as pd
import ast
from datetime import datetime
from .generate_query import multi_process_query
from .post_process import post_process, verify_and_filter_entities, retrieve_wikipedia_pages
from .calculate_stat import calculate_stats, plot_stats, plot_pan_stats


def main():
    # Configuration
    seed_sample_size = 100
    re_sample = False # use the existing dataset
    base_dir = "/mnt/250T_ceph/tristanysui/okgqa"
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamp = datetime.now().strftime('%Y%m%d')
    
    # Generate raw queries
    os.makedirs(os.path.join(base_dir, "queries"), exist_ok=True)
    raw_dataset_path = os.path.join(base_dir, "queries", f"questions_{timestamp}_{seed_sample_size}.csv")
    
    print(f"Generating queries based on {seed_sample_size} seed instructions...")
    if not os.path.exists(raw_dataset_path) or re_sample == True:
        df_raw = multi_process_query(raw_dataset_path, seed_sample_size=seed_sample_size)
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
    retrieve_wikipedia_pages(df_final)
    df_final.to_csv(os.path.join(base_dir, "queries", f"questions_{timestamp}_{seed_sample_size}_post_processed.csv"))
    
    # Calculate statistics
    type_counts, type_naturalness_counts, type_difficulty_counts = calculate_stats(df_final)
    print(type_counts)
    print(type_naturalness_counts)
    print(type_difficulty_counts)
    
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    plot_dir = os.path.join(base_dir, "plots")
    
    # Plot statistics
    plot_stats(type_counts, type_naturalness_counts, type_difficulty_counts, plot_dir)
    plot_pan_stats(type_counts, plot_dir)

if __name__ == "__main__":
    main() 