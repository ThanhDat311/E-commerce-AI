import pandas as pd
import os
import time
import numpy as np

def prepare_balanced_dataset():
    input_csv = r"D:\InformationTechnology\Semester6\project_AI\dataset\archive\rba-dataset.csv"
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, 'balanced_rba.csv')
    
    chunk_size = 1000000 # Read 1M rows at a time
    fraud_keep_rate = 1.0 # Keep 100% of frauds
    normal_keep_rate = 0.005 # Keep 0.5% of normal traffic (approx 5k per 1M rows)
    
    print(f"Starting to process 9GB CSV file: {input_csv}")
    print(f"Output will be saved to: {output_csv}")
    print(f"Configuration: Keep ALL Account Takeovers, Keep {normal_keep_rate*100}% of normal traffic.\n")
    
    start_time = time.time()
    
    total_rows_processed = 0
    total_frauds_kept = 0
    total_normals_kept = 0
    
    # Write header first time
    write_header = True
    
    try:
        # We need to iterate over chunks
        chunk_iter = pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False)
        
        for i, chunk in enumerate(chunk_iter):
            # Find the target column. It might be lowercase or have spaces.
            cols = chunk.columns.tolist()
            target_col = None
            for col in cols:
                if 'account takeover' in str(col).lower():
                    target_col = col
                    break
                    
            if not target_col:
                print("Error: Could not find 'Account Takeover' column in dataset.")
                return
                
            # Classify rows
            # Handle possible boolean or string values
            is_fraud_mask = chunk[target_col].astype(str).str.upper() == 'TRUE'
            
            # Split data
            fraud_rows = chunk[is_fraud_mask]
            normal_rows = chunk[~is_fraud_mask]
            
            # Sample normal rows randomly
            sampled_normal_rows = normal_rows.sample(frac=normal_keep_rate, random_state=42)
            
            # Combine
            filtered_chunk = pd.concat([fraud_rows, sampled_normal_rows])
            
            # Save to output file
            # mode 'a' means append
            filtered_chunk.to_csv(output_csv, mode='a', header=write_header, index=False)
            write_header = False # Only write header on the first iteration
            
            # Update stats
            total_rows_processed += len(chunk)
            total_frauds_kept += len(fraud_rows)
            total_normals_kept += len(sampled_normal_rows)
            
            print(f"Processed Chunk {i+1} ({total_rows_processed:,} total rows). "
                  f"Kept {len(fraud_rows)} frauds & {len(sampled_normal_rows)} normals.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return
        
    print("\n--- Processing Complete ---")
    print(f"Total time: {round(time.time() - start_time, 2)} seconds")
    print(f"Total rows scanned: {total_rows_processed:,}")
    print(f"Final Balanced Dataset size: {total_frauds_kept + total_normals_kept:,} rows "
          f"({total_frauds_kept:,} Frauds, {total_normals_kept:,} Normals)")
          
if __name__ == "__main__":
    prepare_balanced_dataset()
