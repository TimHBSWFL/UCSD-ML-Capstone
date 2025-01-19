import io
import sys
import os
import boto3
import json
import base64
from flask import Flask, request, jsonify, render_template
from PIL import Image
from preprocessing_images import merge_csvs, preprocess_user_images
# from inference_utils import MultimodalClassifierInference

app = Flask(__name__)

# SageMaker endpoint name
SAGEMAKER_ENDPOINT_NAME = "multimodal-classifier-endpoint-011725"

# Initialize inference utility
# inference_util = MultimodalClassifierInference(SAGEMAKER_ENDPOINT_NAME)

# S3 bucket details
S3_BUCKET = "sagemaker-studio-619071335465-8h7owh9eftx"
MAIN_IMAGE_DIR = "samples/image classification/"
EXCLUDED_FILE = "samples/image classification/final_image_to_text_results_other.csv"

# Load and merge CSVs into a DataFrame at startup
csv_df = merge_csvs(S3_BUCKET, MAIN_IMAGE_DIR, EXCLUDED_FILE)

@app.route('/')
def index():
    """
    Serve the index.html file.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check for the uploaded image file
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        # Get the uploaded file
        image_file = request.files['image']

        # Ensure the file has a valid filename
        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Directly use the filename as the photo_id
        photo_id = image_file.filename

        # Process the image (convert to base64 for embedding)
        image = Image.open(image_file)
        print(f"Image mode for {image_file}: {image.mode}")

        if image.mode != "RGB":
            image = image.convert("RGB")


        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        # decoded_image = base64.b64decode(image_base64)
        # image = Image.open(io.BytesIO(decoded_image))

        # Prepare the user_images dictionary
        user_images = [
            {
                "filename": photo_id,
                "image": image_base64   
            }
        ]


        # Preprocess user-uploaded images
        processed_results = preprocess_user_images(user_images, csv_df)

        # Extract processed features
        try:
            image_features = processed_results[0][0]
            text_features = processed_results[1][0]
            label = processed_results[2][0]
            restaurant = processed_results[3][0]
            photo_id = processed_results[4][0]
        except IndexError:
            return jsonify({"error": "Invalid processed_results structure"}), 500

        # print(image_features)
        # print(text_features)
        print(restaurant)

        # Use inference utility to make predictions
        # prediction = inference_util.predict(image_features, text_features)
        # print(prediction)

        image_features = image_features.squeeze()
        text_features = text_features.squeeze()

        print("Image features shape after squeeze:", image_features.shape)
        print("Text features shape after squeeze:", text_features.shape)

        payload = {
            "image_embeddings": image_features.tolist(),
            "text_embeddings": text_features.tolist()
        }

        
        payload_size = sys.getsizeof(json.dumps(payload))
        print(f"Payload size: {payload_size} bytes")

        region="us-east-2"

        client = boto3.client('sagemaker-runtime', region_name=region)

        response = client.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            Body=json.dumps(payload),
            ContentType="application/json"
        )

        result = response['Body'].read().decode('utf-8')
        print(result)

        # Return prediction and metadata
        return jsonify({
            "restaurant": restaurant,
            "photo_id": photo_id,
            "true_label": "Positive" if label == 1 else "Negative",
            "predicted_label": "Positive" if prediction[0] > 0.5 else "Negative",
            "confidence_score": prediction[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
