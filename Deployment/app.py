import os
import json
import boto3
import torch
from PIL import Image
import pandas as pd
from io import BytesIO
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import CLIPProcessor, CLIPModel


# Initialize FastAPI app
app = FastAPI()

# Configure Jinja2Templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Serve static files (optional, for CSS/JS)
app.mount("/temp_images", StaticFiles(directory="temp_images"), name="temp_images")
app.mount("/static", StaticFiles(directory="static"), name="static")


# S3 bucket details and global variables
S3_BUCKET = "sagemaker-studio-619071335465-8h7owh9eftx"
MAIN_IMAGE_DIR = "samples/image classification/"
EXCLUDED_FILE = "samples/image classification/final_image_to_text_results_other.csv"
CLIP_MODEL_PATH = "/home/sagemaker-user/clip_model"
CLIP_PROCESSOR_PATH = "/home/sagemaker-user/clip_processor"
ENDPOINT_NAME = "multimodal-classifier-endpoint-0126257"
endpoint_name_txt = "hf-text-reviews-01044"

# S3 and SageMaker runtime clients
s3 = boto3.client("s3")
runtime_client = boto3.client("sagemaker-runtime")

# Load CLIP model and processor at startup
@app.on_event("startup")
def load_clip_model():
    global clip_model, clip_processor, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_PROCESSOR_PATH)


def fetch_csv_data(bucket: str, prefix: str, excluded_file: str):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    csv_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv') and obj['Key'] != excluded_file]
    
    chunks = []
    for file_key in csv_files:
        obj = s3.get_object(Bucket=bucket, Key=file_key)
        chunk = pd.read_csv(obj['Body'], chunksize=10000)
        chunks.extend(chunk)
    return pd.concat(chunks, ignore_index=True)

def merge_csvs(bucket_name, main_image_dir, excluded_file):
    df_concat = fetch_csv_data(bucket_name, main_image_dir, excluded_file).drop_duplicates(subset='photo_id')
    df_captions = pd.read_csv(f"s3://{bucket_name}/{excluded_file}")
    df_captions['photo_id'] = df_captions['photo_id'].str.replace('.jpg', '', regex=False)
    df_merged = df_captions.merge(df_concat, how='left', on='photo_id')
    df_merged['photo_id'] = df_merged['photo_id'] + '.jpg'
    return df_merged

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Serve the main HTML interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-images/")
async def process_images(user_images: List[UploadFile] = File(...)):
    try:
        # Ensure the temporary directory exists
        os.makedirs("temp_images", exist_ok=True)

        # Save the uploaded image to a temporary directory
        uploaded_image_paths = []
        image_features = []
        text_features = []

        for uploaded_file in user_images:
            # Save the image file to a temporary location
            photo_id = uploaded_file.filename
            temp_image_path = f"temp_images/{photo_id}"
            uploaded_image_paths.append(temp_image_path)

            # Read the image contents and save them to the temporary file
            contents = await uploaded_file.read()
            with open(temp_image_path, "wb") as temp_file:
                temp_file.write(contents)

            image = Image.open(BytesIO(contents))

            # Merge CSVs and match photo_id
            merged_df = merge_csvs(S3_BUCKET, MAIN_IMAGE_DIR, EXCLUDED_FILE)
            matched_row = merged_df.loc[merged_df['photo_id'] == photo_id]
            if matched_row.empty:
                continue

            caption = matched_row['caption'].iloc[0]
            inputs = clip_processor(
                text=[caption], images=image, return_tensors="pt", padding=True
            ).to(device)

            # Get the embeddings for image and text
            with torch.no_grad():
                image_embedding = clip_model.get_image_features(inputs['pixel_values']).squeeze().cpu()
                text_embedding = clip_model.get_text_features(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                ).squeeze().cpu()

            image_features.append(image_embedding)
            text_features.append(text_embedding)

        # Ensure features were extracted
        if image_features and text_features:
            image_features_tensor = torch.stack(image_features)
            text_features_tensor = torch.stack(text_features)

            # Prepare payload for SageMaker endpoint
            payload = {
                "image_embedding": image_features_tensor.tolist(),
                "text_embedding": text_features_tensor.tolist()
            }

            # Invoke the SageMaker endpoint
            response = runtime_client.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(payload),
            )

            # Parse the response
            response_payload = response["Body"].read().decode("utf-8")
            prediction_scores = json.loads(response_payload)

            if isinstance(prediction_scores, list) and len(prediction_scores) > 0:
                prediction_score = float(prediction_scores[0])
                predicted_label = "Upscale" if prediction_score > 0.5 else "Fast Food"
            else:
                raise ValueError("No valid predictions found in the response.")

            # Return the prediction along with the image URL
            response_data = {
                "prediction_score": prediction_score,
                "predicted_label": predicted_label,
                "image_url": f"/temp_images/{uploaded_image_paths[0].split('/')[-1]}"   # Path to the uploaded image
            }

            return response_data

        else:
            raise ValueError("No features extracted for predictions")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

# Define Pydantic model
class TextInput(BaseModel):
    text: List[str]  # Ensures FastAPI receives a valid JSON request

@app.post("/process-text/")
async def process_text(request: TextInput):
    try:
        print("Input text:", request.text)  # Log the text input for debugging
        input_payload = json.dumps({"text": request.text})  # Send as a list of strings
        # Invoke the SageMaker endpoint
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name_txt,
            ContentType="application/json",
            Body=input_payload
        )
        # Read and parse the response
        response_body = json.loads(response["Body"].read().decode("utf-8"))
        sentiments = response_body.get("label", ["Unknown"] * len(request.text))  # Ensure we get one prediction per text
        return {"sentiments": sentiments}
    except Exception as e:
        print(f"Error: {str(e)}")  # Log error for debugging
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

