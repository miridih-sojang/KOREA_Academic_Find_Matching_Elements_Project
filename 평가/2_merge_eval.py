import os
import pandas as pd
import argparse

def calculate_average(model, output_dir):
    model_name = model.split('/')[-1]
    output_dir = output_dir.split('/')[-1]
    folder_path = f'{output_dir}/{model_name}'
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('_with_similarities.csv')]
    
    if not all_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    combined_df_list = []
    
    for file in all_files:
        keyword = os.path.basename(file).replace('_with_similarities.csv', '')
        df = pd.read_csv(file)
        df.insert(0, 'keyword', keyword)
        combined_df_list.append(df)
    
    combined_df = pd.concat(combined_df_list, ignore_index=True)
    
    print("Combined DataFrame:")
    print(combined_df)
    
    # Save the combined DataFrame
    combined_csv_path = f'combined_{model_name.replace(":", "-")}_{output_dir}.csv'
    combined_df.to_csv(combined_csv_path, header=True, index=False, encoding='utf-8-sig')
    print(f"Combined CSV saved to {combined_csv_path}")
    
    average_df = pd.DataFrame(combined_df.mean()).transpose()
    average_df.insert(0, 'keyword', 'overall_average')
    
    print("Averages:")
    print(average_df)
    
    average_csv_path = f'result_{model_name.replace(":", "-")}_{output_dir}.csv'
    average_df.to_csv(average_csv_path, header=True, index=False, encoding='utf-8-sig')
    print(f"Average CSV saved to {average_csv_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate averages from CSV files in a model folder.')
    parser.add_argument('model', type=str, help='The model name to calculate averages for')
    parser.add_argument('output_dir', type=str, help='Output directory path')
    
    args = parser.parse_args()
    calculate_average(args.model, args.output_dir)
    
    
    