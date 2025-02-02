import os
import io
import boto3
import sagemaker
import torch
from torch import nn
import numpy as np
import datetime as dt
from io import BytesIO
from sklearn.metrics import accuracy_score
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from PIL import Image


s3 = boto3.client('s3')

def extract_features_from_images_text(
    sample: int,
    bucket_name: str,
    main_image_dir: str,
    excluded_file: str,
    image_dir: str,
    clip_model_path: str,
    clip_processor_path: str
):


    device = "cuda" if torch.cuda.is_available() else "cpu"


    def fetch_csv_data(bucket: str, prefix: str):
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        csv_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv') and obj['Key'] != excluded_file]

        chunks = []
        for file_key in csv_files:
            file_path = f"s3://{bucket}/{file_key}"
            for chunk in pd.read_csv(file_path, chunksize=10000):
                chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    df_concat = fetch_csv_data(bucket_name, main_image_dir).drop_duplicates(subset='photo_id')
    df_captions = pd.read_csv(f"s3://{bucket_name}/{excluded_file}")
    df_captions['photo_id'] = df_captions['photo_id'].str.replace('.jpg', '', regex=False)
    df_merged = df_captions.merge(df_concat, how='left', on='photo_id')
    df_merged['photo_id'] = df_merged['photo_id'] + '.jpg'

    sampled_df = df_merged.sample(n=sample, replace=False)
    sampled_df['labels'] = sampled_df['labels'].map({"fast food": 0, "fine dining": 1})

    # restaurant_name = sampled_df.iloc[0]['name']
    # photo_info = sampled_df.iloc[0]['photo_id']

    def get_image_paths(bucket: str, prefix: str):
        image_keys = []
        continuation_token = None
        while True:
            params = {'Bucket': bucket, 'Prefix': prefix}
            if continuation_token:
                params['ContinuationToken'] = continuation_token
            response = s3.list_objects_v2(**params)
            image_keys += [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.jpg')]
            if not response.get('IsTruncated'):
                break
            continuation_token = response.get('NextContinuationToken')
        return image_keys

    image_paths = get_image_paths(bucket_name, image_dir)


    clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_processor_path)


    def fetch_image(bucket: str, key: str):
        response = s3.get_object(Bucket=bucket, Key=key)
        return Image.open(BytesIO(response['Body'].read())).convert("RGB")


    image_features, text_features, labels, restaurants, photo_ids = [], [], [], [], []
    for idx, row in sampled_df.iterrows():
        image_key = next((path for path in image_paths if os.path.basename(path) == row["photo_id"]), None)
        if image_key:
            image = fetch_image(bucket_name, image_key)
            inputs = clip_processor(text=[row["caption"]], images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                image_embedding = clip_model.get_image_features(pixel_values=inputs['pixel_values']).squeeze().cpu()
                text_embedding = clip_model.get_text_features(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                ).squeeze().cpu()
            image_features.append(image_embedding)
            text_features.append(text_embedding)
            labels.append(row['labels'])
            restaurants.append(row["name"])
            photo_ids.append(row["photo_id"])
        else:
            print(f"Image {row['photo_id']} not found in S3.")

        

    return torch.stack(image_features), torch.stack(text_features), torch.tensor(labels), restaurants, photo_ids