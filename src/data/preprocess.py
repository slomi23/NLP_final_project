import pandas as pd
import json
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_jsonl_to_dataframe(jsonl_file_path, encoding='utf-8'):
    data = []
    try:
        with open(jsonl_file_path, 'r', encoding=encoding) as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON line {line_num} in {jsonl_file_path}: {e}")
                    continue
    
    except UnicodeDecodeError:
        logger.warning(f"Failed to decode {jsonl_file_path} with {encoding}, trying 'latin-1'")
        try:
            with open(jsonl_file_path, 'r', encoding='latin-1') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON line {line_num} in {jsonl_file_path}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Could not read file {jsonl_file_path}: {e}")
            return pd.DataFrame()
    
    logger.info(f"Loaded {len(data)} records from {jsonl_file_path}")
    return pd.DataFrame(data)


def load_all_jsonl_files(directory_path, encoding='utf-8'):
    directory = Path(directory_path)
    if not directory.exists():
        logger.error(f"Directory {directory_path} does not exist")
        return pd.DataFrame()
    
    jsonl_files = list(directory.glob('*.jsonl'))
    
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {directory_path}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(jsonl_files)} JSONL files in {directory_path}")
    
    all_dataframes = []
    
    for jsonl_file in jsonl_files:
        df = load_jsonl_to_dataframe(jsonl_file, encoding)
        if not df.empty:
            df['source_file'] = str(jsonl_file.name)
            all_dataframes.append(df)
    
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"Combined DataFrame shape: {combined_df.shape}")
        return combined_df
    else:
        logger.error("No valid data loaded from JSONL files")
        return pd.DataFrame()

def explore_dataframe(df):
    print(f"\nDataFrame Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nSample values from key columns:")
    for col in ['title', 'summary'] if 'title' in df.columns else df.columns[:3]:
        if col in df.columns:
            print(f"\n{col} sample values:")
            print(df[col].dropna().head(3))

def clean_dataframe(df):
    cleaned_df = df.copy()
    text_columns = ['summary', 'abstract', 'content']
    available_text_cols = [col for col in text_columns if col in cleaned_df.columns]
    
    if available_text_cols:
        cleaned_df = cleaned_df.dropna(subset=available_text_cols, how='all')
        cleaned_df['combined_text'] = cleaned_df[available_text_cols].fillna('')
        
        if 'title' in cleaned_df.columns:
            cleaned_df['combined_text'] = cleaned_df['title'].fillna('') + ' ' + cleaned_df['combined_text']

    if 'combined_text' in cleaned_df.columns:
        cleaned_df['word_count'] = cleaned_df['combined_text'].str.split().str.len()

    if 'word_count' in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df['word_count'] >= 10]
    
    logger.info(f"Cleaned DataFrame shape: {cleaned_df.shape}")
    return cleaned_df

if __name__ == "__main__":
    data_directory = "data/raw"
    logger.info("Loading JSONL files...")
    df = load_all_jsonl_files(data_directory)
    
    if not df.empty:
        explore_dataframe(df)
        logger.info("Cleaning DataFrame...")
        cleaned_df = clean_dataframe(df)
        output_file = "data/processed/search_corpus.csv"
        logger.info(f"Saving processed data to {output_file}")
        cleaned_df.to_csv(output_file, index=False)
        
        print(f"\nSuccessfully processed and saved {len(cleaned_df)} records to {output_file}")
    else:
        print("No data was loaded. Please check your JSONL files and directory path.")