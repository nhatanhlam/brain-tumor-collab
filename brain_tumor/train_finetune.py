from fastai.vision.all import *
import timm
import torch
import json
from pathlib import Path

def safe_save_model(learner, filename):
    """Lưu mô hình an toàn để tránh lỗi CUDA khi pickling."""
    device = next(learner.model.parameters()).device
    learner.model.cpu()  # Chuyển về CPU trước khi lưu
    learner.export(f"models/{filename}.pkl")  # Lưu mô hình dưới dạng .pkl
    learner.model.to(device)  # Chuyển lại về thiết bị ban đầu

def evaluate_model(learner, test_path):
    """Chạy mô hình trên tập test và tính độ chính xác của từng nhãn."""
    test_files = get_image_files(test_path)
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

    return accuracies

def main():
    Path('models').mkdir(exist_ok=True)

    # Định nghĩa đường dẫn dữ liệu
    data_path = Path('data/Training')
    test_path = Path('data/Testing')

    # Kiểm tra xem các thư mục dữ liệu có tồn tại không
    assert all((data_path/cls).exists() for cls in ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']), \
           "Thiếu một hoặc nhiều thư mục dữ liệu!"

    # Tạo DataBlock với Augmentation
    brain_tumor = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(
            do_flip=True, flip_vert=True, max_rotate=30.0, 
            max_zoom=1.2, max_lighting=0.2, max_warp=0.2
        )
    )

    # Tạo DataLoaders
    dls = brain_tumor.dataloaders(data_path, bs=32, num_workers=0)

    # Khởi tạo mô hình ResNet101
    learn = vision_learner(dls, 'resnet101', metrics=accuracy)
    learn.to_fp16()  # Sử dụng mixed precision để tăng tốc
    learn.model_dir = Path('models')

    # Huấn luyện với Fine-Tuning
    print("🔄 Huấn luyện mô hình với Fine-Tuning...")
    learn.fine_tune(15, base_lr=3e-4)

    # Biến để theo dõi mô hình tốt nhất
    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    epoch = 0
    max_epochs = 100

    # Lưu log training
    log_file = Path("models/training_log.txt")
    with open(log_file, "w") as log:
        log.write("Epoch | Train Loss | Test Loss | Train Acc | Test Acc\n")
        log.write("-" * 50 + "\n")

        while epoch < max_epochs:
            epoch += 1
            print(f"Epoch {epoch} bắt đầu...")

            learn.fit_one_cycle(1, 3e-3)

            train_loss = learn.recorder.losses[-1].item()
            train_acc = learn.recorder.values[-1][1]

            # Đánh giá trên tập test
            test_acc_dict = evaluate_model(learn, test_path)
            test_acc = sum(test_acc_dict.values()) / len(test_acc_dict)  # Trung bình accuracy
            test_loss = None  # Không có test loss vì fastai không tính

            train_loss_str = f"{train_loss:.4f}"
            test_loss_str = "N/A"
            train_acc_str = f"{train_acc:.4f}"
            test_acc_str = f"{test_acc:.4f}"

            print(f"Epoch {epoch} - Train Loss: {train_loss_str}, Test Acc: {test_acc_str}")

            log.write(f"{epoch:<5} | {train_loss_str} | {test_loss_str} | {train_acc_str} | {test_acc_str}\n")

            if test_acc > best_val_acc:
                best_val_acc = test_acc
                print("📥 Lưu mô hình tốt nhất...")
                safe_save_model(learn, 'best_model')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("⏹️ Dừng training do không cải thiện liên tiếp.")
                break

    print("💾 Lưu mô hình cuối cùng...")
    safe_save_model(learn, 'last_model')

    # Đánh giá mô hình tốt nhất sau training
    print("🔍 Đánh giá mô hình tốt nhất trên tập test...")
    best_model = load_learner("models/best_model.pkl")
    final_test_acc_dict = evaluate_model(best_model, test_path)

    # Lưu kết quả test vào JSON
    with open("models/test_results.json", "w") as f:
        json.dump(final_test_acc_dict, f, indent=4)

    print("✅ Kết quả test từng nhãn:")
    for label, acc in final_test_acc_dict.items():
        print(f"🔹 {label}: {acc:.2f}%")

if __name__ == '__main__':
    main()
