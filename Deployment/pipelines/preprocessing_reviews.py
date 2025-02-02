import boto3
import pandas as pd
import random


s3 = boto3.client('s3')

def preprocess_text_reviews(bucket_name: str, sample_size: int) -> pd.DataFrame:

    main_text_dir = 'samples/text reviews/'
  
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=main_text_dir)
    parquet_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.parquet')]

    if len(parquet_files) == 1:
        parquet_file_key = parquet_files[0]
        print(f"Found Parquet file: {parquet_file_key}")
    else:
        raise ValueError(f"Expected exactly one Parquet file, but found {len(parquet_files)}")

    s3_uri = f"s3://{bucket_name}/{parquet_file_key}"

    df_reviews = pd.read_parquet(s3_uri, columns=["text", "stars_reviews"])
    sampled_df = df_reviews.sample(n=sample_size, replace=False)

    print(f"Sampled {sample_size} rows from {parquet_file_key}.")

    return sampled_df
