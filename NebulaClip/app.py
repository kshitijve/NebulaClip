import os
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
import base64
from PIL import Image
import io
import sys
from transformers import AutoModelForImageSegmentation
from pathlib import Path
from torchvision import transforms

app = Flask(__name__, static_folder='.', static_url_path='')

def load_model():
    """Load the background removal model."""
    try:
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        model_id = 'briaai/RMBG-2.0'
        
        # Check if model files exist locally
        model_file = os.path.join(model_dir, "model.safetensors")
        config_file = os.path.join(model_dir, "config.json")
        
        if os.path.exists(model_file) and os.path.exists(config_file):
            model = AutoModelForImageSegmentation.from_pretrained(
                model_dir,
                trust_remote_code=True
            )
            
            # Move model to appropriate device and set to eval mode
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            return model
        else:
            return None
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Initialize model globally after the function is defined
model = load_model()
if model is None:
    logger.error("Failed to load model. Application may not work correctly.")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/remove-bg', methods=['POST'])
def remove_background():
    try:
        # Get image from request
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image provided'}), 400

        # Setup image transformation
        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Read and transform image
        img = Image.open(file.stream).convert('RGB')
        original_size = img.size
        input_tensor = transform_image(img).unsqueeze(0)

        # Move to same device as model
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Process with model
        with torch.no_grad():
            output = model(input_tensor)
            # Get the last output and apply sigmoid
            pred = output[-1].sigmoid().cpu()
            
            # Convert prediction to mask
            pred_mask = pred[0].squeeze()
            mask_pil = transforms.ToPILImage()(pred_mask)
            mask_pil = mask_pil.resize(original_size)

        # Create final image with transparency
        result = img.copy()
        result.putalpha(mask_pil)

        # Convert to base64
        buffered = io.BytesIO()
        result.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({'image': img_str})

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure model is loaded before starting the server
    if model is None:
        logger.error("Model failed to load. Please check model files and try again.")
        sys.exit(1)
    
    logger.info("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)