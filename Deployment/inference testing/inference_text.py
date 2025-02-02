import os
import json
import boto3
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_dir = '/home/sagemaker-user/checkpoint-29650'
bucket_name = "sagemaker-studio-619071335465-8h7owh9eftx"
s3_key = "training/outputs/checkpoint-29650"
s3 = boto3.client('s3')

def load_model_and_tokenizer(local_dir, s3_key, bucket_name):
    if not os.path.exists(local_dir) or not all(
        os.path.exists(os.path.join(local_dir, file_name)) for file_name in ["config.json", "model.safetensors"]
    ):
        os.makedirs(local_dir, exist_ok=True)
        for file_name in ["config.json", "model.safetensors"]:
            s3.download_file(bucket_name, f"{s3_key}/{file_name}", os.path.join(local_dir, file_name))

    model = AutoModelForSequenceClassification.from_pretrained(local_dir)
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    return model, tokenizer

def predict_fn(input_data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    # Input validation
    if not isinstance(input_data, list) or not all(isinstance(item, str) for item in input_data):
        raise ValueError("Expected input_data to be a list of strings.")

    inputs = tokenizer(input_data, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(dim=-1)
    return predictions.cpu().numpy().tolist()

def output_fn(predictions, content_type):
    label_map = {0: "Negative", 1: "Positive"}
    prediction_labels = [label_map[prediction] for prediction in predictions]
    return {"label": prediction_labels}

def input_fn(request_body, content_type):
    if content_type == "application/json":
        try:
            parsed_body = json.loads(request_body)
            if "text" not in parsed_body:
                raise ValueError("Missing 'text' key in input JSON.")
            return parsed_body["text"]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def model_fn(model_dir, context=None):
    model, tokenizer = load_model_and_tokenizer(model_dir, s3_key, bucket_name)
    return model, tokenizer
