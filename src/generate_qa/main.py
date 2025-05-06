import os
import pandas as pd
from datetime import datetime
from .generate_query import multi_process_query
from .post_process import post_process, verify_and_filter_entities
from .calculate_stat import calculate_stats, plot_stats, plot_pan_stats


def main():
    # Configuration
    sample_size = 100   
    base_dir = "/mnt/250T_ceph/tristanysui/okgqa"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate raw queries
    raw_dataset_path = os.path.join(base_dir, f"questions_{timestamp}_{sample_size}.csv")
    print(f"Generating {sample_size*5} queries...")
    df_raw = multi_process_query(raw_dataset_path, sample_size=sample_size)
    
    # Post-process the generated queries
    print("\nPost-processing queries...")
    df_post_processed = post_process(df_raw)
    
    # Verify and filter entities
    print("\nVerifying and filtering entities...")
    df_final = verify_and_filter_entities(df_post_processed)
    
    # Save final results
    print(f"Total queries generated: {len(df_raw)}")
    print(f"Queries after post-processing: {len(df_post_processed)}")
    print(f"Final valid queries: {len(df_final)}")
    
    # Calculate statistics
    type_counts, type_naturalness_counts, type_difficulty_counts = calculate_stats(df_final)
    
    # Plot statistics
    plot_stats(type_counts, type_naturalness_counts, type_difficulty_counts)
    plot_pan_stats(type_counts)

if __name__ == "__main__":
    main() 