from fastai.vision.all import *
import os
import sys

# Định nghĩa đường dẫn đến mô hình đã lưu (best_model.pkl)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models/best_model.pkl')

def load_model():
    """Tải mô hình đã huấn luyện từ file .pkl"""
    return load_learner(model_path)

# Định nghĩa mapping từ nhãn dự đoán sang tumor_id
# Lưu ý: Mapping này phải nhất quán với hệ thống gợi ý bệnh viện (predict_hospital.py)
# Ví dụ:
#   "glioma_tumor"      -> tumor_id = 1
#   "meningioma_tumor"  -> tumor_id = 2
#   "no_tumor"          -> tumor_id = 3  (không cần gợi ý)
#   "pituitary_tumor"   -> tumor_id = 4
tumor_id_mapping = {
    "glioma_tumor": 1,
    "meningioma_tumor": 2,
    "no_tumor": 3,
    "pituitary_tumor": 4
}

if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        print("Usage: python predict.py image_path")
        sys.exit(1)

    # Load mô hình chẩn đoán
    learner = load_model()

    # Tạo ảnh để dự đoán
    img = PILImage.create(img_path)
    pred, pred_idx, probs = learner.predict(img)

    # Lấy nhãn dự đoán dạng chuỗi và tra cứu tumor_id tương ứng
    pred_str = str(pred)
    tumor_id = tumor_id_mapping.get(pred_str, None)

    # In kết quả dự đoán và tumor_id (tumor_id này có thể dùng để gọi thông tin gợi ý bệnh viện)
    print(f"Prediction: {pred_str}")
    print(f"Tumor ID: {tumor_id}")
    print(f"Probability: {probs[pred_idx]:.4f}")
