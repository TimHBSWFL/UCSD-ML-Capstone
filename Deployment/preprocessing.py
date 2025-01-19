import os
import boto3
import torch
from torch import nn
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO

# S3 bucket details
S3_BUCKET = "sagemaker-studio-619071335465-8h7owh9eftx"
MAIN_IMAGE_DIR = "samples/image classification/"
EXCLUDED_FILE = "samples/image classification/final_image_to_text_results_other.csv"
CLIP_MODEL_PATH = "/home/sagemaker-user/clip_model"
CLIP_PROCESSOR_PATH = "/home/sagemaker-user/clip_processor"

# Initialize S3 client
s3 = boto3.client('s3')

def fetch_csv_data(bucket: str, prefix: str, excluded_file: str):
    """
    Fetch and concatenate CSV data from S3, excluding a specific file.
    """
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    csv_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv') and obj['Key'] != excluded_file]
    
    chunks = []
    for file_key in csv_files:
        file_path = f"s3://{bucket}/{file_key}"
        for chunk in pd.read_csv(file_path, chunksize=10000):
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def merge_csvs(bucket_name, main_image_dir, excluded_file):
    """
    Merge CSVs into a single DataFrame, aligning with photo IDs and captions.
    """
    df_concat = fetch_csv_data(bucket_name, main_image_dir, excluded_file).drop_duplicates(subset='photo_id')
    df_captions = pd.read_csv(f"s3://{bucket_name}/{excluded_file}")
    df_captions['photo_id'] = df_captions['photo_id'].str.replace('.jpg', '', regex=False)
    df_merged = df_captions.merge(df_concat, how='left', on='photo_id')
    df_merged['photo_id'] = df_merged['photo_id'] + '.jpg'
    return df_merged

def preprocess_user_images(user_images, merged_df):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_PROCESSOR_PATH)

    image_features, text_features, labels, restaurants, photo_ids = [], [], [], [], []

    for item in user_images:
        try:
            # Extract the photo_id
            photo_id = item['filename']
            print(f"Debug: Processing photo_id {photo_id}")

            # Validate the image
            image = item['image']
            if not isinstance(image, Image.Image):
                print(f"Debug: Invalid image for photo_id {photo_id}")
                continue

            # Match photo_id in merged_df
            matched_row = merged_df.loc[merged_df['photo_id'] == photo_id]
            if matched_row.empty:
                print(f"Debug: No match for photo_id {photo_id}")
                continue

            # Extract metadata
            caption = matched_row['caption'].iloc[0]
            label = matched_row['labels'].iloc[0]
            restaurant = matched_row['name'].iloc[0]

            # Process with CLIP
            inputs = clip_processor(
                text=[caption], images=image, return_tensors="pt", padding=True
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                image_embedding = clip_model.get_image_features(pixel_values=inputs['pixel_values']).squeeze().cpu()
                text_embedding = clip_model.get_text_features(
                    input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']
                ).squeeze().cpu()

            # Append results
            image_features.append(image_embedding)
            text_features.append(text_embedding)
            labels.append(label)
            restaurants.append(restaurant)
            photo_ids.append(photo_id)

        except Exception as e:
            print(f"Error processing image {photo_id}: {e}")
            continue

    # Return results
    return (
        torch.stack(image_features) if image_features else torch.empty(0),
        torch.stack(text_features) if text_features else torch.empty(0),
        labels,
        restaurants,
        photo_ids,
    )




def process_images_from_s3(bucket_name, main_image_dir, excluded_file, user_images):
    """
    Combine logic for processing S3 images and user-uploaded images.
    """
    # Fetch and merge CSV data
    merged_df = merge_csvs(bucket_name, main_image_dir, excluded_file)

    # Preprocess user-uploaded images
    user_image_features, user_text_features, user_labels, user_restaurants, user_photo_ids = preprocess_user_images(user_images, merged_df)
    
    return {
        "user_image_features": user_image_features,
        "user_text_features": user_text_features,
        "user_labels": user_labels,
        "user_restaurants": user_restaurants,
        "user_photo_ids": user_photo_ids
    }
