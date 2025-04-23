import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np

# Define class names
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=4)

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models_old", "best_vit_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            f"Ensure the model has been trained and saved in the 'models' directory."
        )
    try:
        # Load fine-tuned weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model checkpoint: {str(e)}")

def predict_class(image_path):
    """Dự đoán lớp của một ảnh"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Raw logits
        pred_idx = logits.argmax(dim=-1).item()  # Lấy chỉ số lớp có xác suất cao nhất
    return CLASS_NAMES[pred_idx]  # Trả về nhãn dự đoán

def evaluate_testing_directory(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    correct_predictions = {label: 0 for label in CLASS_NAMES}
    total_images = {label: 0 for label in CLASS_NAMES}

    # Duyệt qua từng thư mục chứa ảnh test theo nhãn
    for true_label in CLASS_NAMES:
        label_dir = os.path.join(directory_path, true_label)
        if not os.path.exists(label_dir):
            print(f"⚠️ Warning: Directory for label '{true_label}' not found. Skipping.")
            continue

        image_files = [
            os.path.join(label_dir, file)
            for file in os.listdir(label_dir)
            if file.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        total_images[true_label] = len(image_files)

        print(f"🔍 Đang kiểm tra nhãn: {true_label} - Tổng số ảnh: {len(image_files)}")

        for image_path in image_files:
            try:
                predicted_label = predict_class(image_path)
                if predicted_label == true_label:
                    correct_predictions[true_label] += 1
            except Exception as e:
                print(f"❌ Lỗi khi dự đoán {image_path}: {e}")

    # Tính toán tỷ lệ phần trăm đúng của từng nhãn
    print("\n✅ **Kết quả đánh giá mô hình trên tập Test:**")
    for label in CLASS_NAMES:
        if total_images[label] > 0:
            accuracy = (correct_predictions[label] / total_images[label]) * 100
            print(f"🔹 {label}: {accuracy:.2f}% ({correct_predictions[label]}/{total_images[label]})")
        else:
            print(f"⚠️ Không có ảnh nào trong thư mục {label}, bỏ qua.")

if __name__ == "__main__":
    testing_directory = os.path.join(os.path.dirname(__file__), "data", "Testing")
    try:
        load_model()  # Load the model
        evaluate_testing_directory(testing_directory)  # Đánh giá mô hình
    except Exception as e:
        print(f"❌ Error: {e}")
