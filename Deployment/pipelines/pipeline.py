import os
import boto3
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pipelines.preprocessing_reviews import preprocess_text_reviews
from pipelines.inference_reviews import load_model_and_tokenizer, run_inference, postprocess_results


def run_pipeline(num_samples):

    bucket_name = "sagemaker-studio-619071335465-8h7owh9eftx"

    sampled_df = preprocess_text_reviews(bucket_name, num_samples)

    s3_client = boto3.client('s3')
    s3_key = "training/outputs/checkpoint-29650"
    local_dir = "/home/sagemaker-user/checkpoint-29650"
    
    model, tokenizer = load_model_and_tokenizer(local_dir, s3_key, bucket_name)

    text_list = sampled_df['text'].tolist()
    reviews_list = sampled_df['stars_reviews'].tolist()

    predictions = run_inference(model, tokenizer, text_list)

    results = postprocess_results(text_list, predictions, reviews_list)

    for result in results:
        print(f"Review: {result['review']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Actual: {result['actual']}")
        print("-" * 40)