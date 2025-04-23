from fastai.vision.all import *
import torch
import json
from pathlib import Path

def evaluate_model(learner, test_path):
    """Chạy mô hình trên tập test và tính độ chính xác của từng nhãn."""
    print("\n🔍 Đánh giá mô hình trên tập Test...")

    test_files = get_image_files(test_path)
    if not test_files:
        print("⚠️ Không tìm thấy ảnh nào trong tập test!")
        return

    test_dl = learner.dls.test_dl(test_files)
    preds, _ = learner.get_preds(dl=test_dl)
    labels = [learner.dls.vocab[i] for i in preds.argmax(dim=1)]

    # Đếm số lượng dự đoán đúng trên mỗi nhãn
    true_counts = {label: 0 for label in learner.dls.vocab}
    total_counts = {label: 0 for label in learner.dls.vocab}

    for file, label in zip(test_files, labels):
        actual_label = file.parent.name
        total_counts[actual_label] += 1
        if actual_label == label:
            true_counts[label] += 1

    # Tính accuracy của từng nhãn
    accuracies = {label: (true_counts[label] / total_counts[label] * 100) if total_counts[label] > 0 else 0 for label in learner.dls.vocab}

    # In kết quả
    for label, acc in accuracies.items():
        print(f"✅ Nhãn: {label} - Độ chính xác: {acc:.2f}% ({true_counts[label]}/{total_counts[label]})")

    # Lưu kết quả vào file JSON
    results_path = test_path.parent / "models" / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(accuracies, f, indent=4)

def main():
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / 'models'
    test_path = base_dir / 'data' / 'Testing'

    # Load model tốt nhất
    learner = load_learner(model_dir / 'best_model.pkl')

    # Đánh giá mô hình trên tập test
    evaluate_model(learner, test_path)

if __name__ == '__main__':
    main()
