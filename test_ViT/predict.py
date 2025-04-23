import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os

# Define class names
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=4)

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_vit_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Ensure the model has been trained and saved in the 'models' directory."
        )
    try:
        # Load fine-tuned weights
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model checkpoint: {str(e)}")

def predict(image_path):
    # Load the model if not already loaded
    load_model()

    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits to probabilities

        print(f"Logits: {logits}")  # Debug logits
        print(f"Probabilities: {probabilities}")

        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0, predicted_class].item()  # Confidence score for predicted class

    return CLASS_NAMES[predicted_class], confidence

if __name__ == "__main__":
    test_image_path = "path/to/test_image.jpg"  # Replace with a valid image path
    try:
        prediction, confidence = predict(test_image_path)
        print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error: {e}")
