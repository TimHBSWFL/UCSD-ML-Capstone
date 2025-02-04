import os
import torch
import json
from torch import nn

# Define the model class
class MultimodalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(MultimodalClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, image_embedding, text_embedding):
        # Concatenate image and text embeddings and pass through the network
        x = torch.cat((image_embedding, text_embedding), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# Define device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
input_dim = 512  # Dimensionality of the embedding vectors
hidden_dim = 256  # Hidden layer size
dropout_rate = 0.45584099758281393  # Dropout rate for regularization

# model_fn that loads the model from the model_dir only once per container
def model_fn(model_dir):
    # Initialize the model and load weights
    model = MultimodalClassifier(input_dim, hidden_dim, dropout_rate).to(device)
    
    # Load model weights from the provided model directory
    model_path = os.path.join(model_dir, 'multimodal_classifier_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode

    return model

# input_fn to handle the incoming request data
def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        inputs = json.loads(request_body)
        # Convert JSON values to PyTorch tensors
        image_embedding = torch.tensor(inputs['image_embedding'], dtype=torch.float32).to(device)
        text_embedding = torch.tensor(inputs['text_embedding'], dtype=torch.float32).to(device)
        return image_embedding, text_embedding
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# predict_fn to run inference with the model
def predict_fn(input_data, model):
    # Run inference with the model
    image_embedding, text_embedding = input_data
    with torch.no_grad():
        output = model(image_embedding, text_embedding)
        
    # Return the prediction as a list (convert tensor to list)
    return output.squeeze(0).tolist()
