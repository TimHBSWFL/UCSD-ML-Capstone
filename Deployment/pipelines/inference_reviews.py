import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

def run_inference(model, tokenizer, text_list):
    inputs = tokenizer(text_list, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    return predictions

def postprocess_results(text_list, predictions, reviews_list):
    results = []
    for review, prediction, actual in zip(text_list, predictions, reviews_list):
        label = "Positive" if prediction.item() == 1 else "Negative"
        results.append({"review": review, "prediction": label, "actual": actual})
    return results

