from fastai.vision.all import *
import timm
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.nn import CrossEntropyLoss
import json
from fastai.data.transforms import IndexSplitter

def safe_save_model(learner, filename, model_dir):
    """Lưu mô hình an toàn để tránh lỗi CUDA khi pickling."""
    device = next(learner.model.parameters()).device
    learner.model.cpu()
    learner.export(model_dir / f"{filename}.pkl")
    learner.model.to(device)

def plot_training_history(train_losses, test_losses, train_accs, test_accs, save_path):
    """Vẽ biểu đồ Accuracy và Loss của Train và Test."""
    epochs = range(1, len(train_losses) + 1)

    # Accuracy Plot
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_accs, label='Train Accuracy', color='blue')
    plt.plot(epochs, test_accs, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(save_path / "accuracy_plot.png")
    plt.close()

    # Loss Plot
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid()
    plt.savefig(save_path / "loss_plot.png")
    plt.close()

def evaluate_model(learner, test_path):
    """Chạy mô hình trên tập test và tính độ chính xác của từng nhãn."""
    test_files = get_image_files(test_path)
    test_dl = learner.dls.test_dl(test_files)
    preds, _ = learner.get_preds(dl=test_dl)
    labels = [learner.dls.vocab[i] for i in preds.argmax(dim=1)]

    true_counts = {label: 0 for label in learner.dls.vocab}
    total_counts = {label: 0 for label in learner.dls.vocab}

    for file, label in zip(test_files, labels):
        actual_label = file.parent.name
        total_counts[actual_label] += 1
        if actual_label == label:
            true_counts[label] += 1

    accuracies = {
        label: (true_counts[label] / total_counts[label] * 100) if total_counts[label] > 0 else 0 
        for label in learner.dls.vocab
    }
    return accuracies

def main():
    # Xác định thư mục gốc
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / 'data' / 'Training'
    test_path = base_dir / 'data' / 'Testing'
    model_dir = base_dir / 'models'
    model_dir.mkdir(exist_ok=True)

    required_dirs = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
    if not all((data_path / cls).exists() for cls in required_dirs):
        print(f"⚠️ Thiếu thư mục dữ liệu! Kiểm tra lại: {required_dirs}")
        return

    # Tạo DataBlock cho training/validation
    brain_tumor = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(
            do_flip=True, flip_vert=True, max_rotate=40,
            max_zoom=1.4, max_lighting=0.3, max_warp=0.4
        )
    )

    # Load DataLoaders cho training
    dls = brain_tumor.dataloaders(data_path, bs=32, num_workers=0)
    print(f"Số ảnh trong train: {len(dls.train_ds)}, validation: {len(dls.valid_ds)}")

    # Khởi tạo mô hình EfficientNet-B3
    learn = vision_learner(dls, "efficientnet_b3", metrics=accuracy, pretrained=True)
    learn.to_fp16()
    learn.model_dir = model_dir

    # Sử dụng trọng số cho loss, ưu tiên cho glioma
    weights = torch.tensor([1.5, 1.0, 1.0, 1.2]).cuda()
    learn.loss_func = CrossEntropyLoss(weight=weights)

    print("🔄 Huấn luyện mô hình với Fine-Tuning...")

    best_test_acc = 0
    patience = 15
    patience_counter = 0
    epoch = 0
    max_epochs = 200

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    log_file = model_dir / "training_log.txt"

    with open(log_file, "w") as log:
        log.write("Epoch | Train Loss | Test Loss | Train Acc | Test Acc\n")
        log.write("-" * 50 + "\n")

        # Tạo DataLoader cho tập test có nhãn bằng DataBlock (sử dụng IndexSplitter([])) để tính test loss
        test_block = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=IndexSplitter([]),  # Tất cả ảnh vào tập "train" => labeled
            get_y=parent_label,
            item_tfms=Resize(224)
        )
        test_dls = test_block.dataloaders(test_path, bs=32, num_workers=0)
        test_dl = test_dls.train  # Sử dụng phần train vì chứa (input, target)

        while epoch < max_epochs:
            epoch += 1
            print(f"Epoch {epoch} bắt đầu...")

            learn.fit_one_cycle(1, 3e-4)

            # 1) Lấy train loss từ recorder
            train_loss = learn.recorder.losses[-1].item()

            # 2) Lấy "val_acc" cũ (nhưng ta không động chạm)
            #    Lưu ý: learn.recorder.values[-1][1] thực chất là validation accuracy
            #    nhưng ta giữ nguyên tên "train_acc" để không ảnh hưởng logic cũ
            train_acc = learn.recorder.values[-1][1]

            # 3) Tính train_acc_real (training accuracy thật sự)
            #    => minimal change: ta không in ra console, chỉ dùng để vẽ
            train_loss_real, train_acc_real = learn.validate(dl=learn.dls[0])
            
            # 4) Tính test loss + test acc qua validate
            val_results = learn.validate(dl=test_dl)
            test_loss_val = val_results[0]
            test_acc_val = val_results[1]
            test_loss = test_loss_val
            test_acc = test_acc_val.item() if hasattr(test_acc_val, "item") else test_acc_val

            # 5) Lưu train_loss thực sự, test_loss
            #    + train_acc_real, test_acc => vẽ biểu đồ
            train_losses.append(train_loss_real)  # train_loss_real
            test_losses.append(test_loss)
            train_accs.append(train_acc_real)     # train_acc_real
            test_accs.append(test_acc)

            # 6) In ra console GIỮ NGUYÊN logic cũ => in Train Loss (thật ra recorder) & Test Acc
            train_loss_str = f"{train_loss:.4f}"
            test_acc_str = f"{test_acc:.4f}"
            print(f"Epoch {epoch} - Train Loss: {train_loss_str}, Test Acc: {test_acc_str}")

            # 7) Ghi log => ta vẫn ghi test_loss (thật) + train_acc (recorder)
            #    Để minimal change, ta giữ "train_acc" cũ, test_loss_val vẽ ra
            log.write(f"{epoch:<5} | {train_loss_str} | {test_loss_val:.4f} | {train_acc:.4f} | {test_acc_str}\n")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                print("📥 Lưu mô hình tốt nhất...")
                safe_save_model(learn, 'best_model', model_dir)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("⏹️ Dừng training do không cải thiện liên tiếp.")
                break

    print("💾 Lưu mô hình cuối cùng...")
    safe_save_model(learn, 'last_model', model_dir)

    # Vẽ biểu đồ (2 đường cho Loss, 2 đường cho Accuracy) => do ta đã có train_losses, test_losses, train_accs, test_accs
    plot_training_history(train_losses, test_losses, train_accs, test_accs, model_dir)

    # Đánh giá mô hình tốt nhất trên tập test
    print("🔍 Đánh giá mô hình tốt nhất trên tập test...")
    best_model = load_learner(model_dir / "best_model.pkl")
    final_test_acc_dict = evaluate_model(best_model, test_path)

    with open(model_dir / "test_results.json", "w") as f:
        json.dump(final_test_acc_dict, f, indent=4)

    print("✅ Kết quả test từng nhãn:")
    for label, acc in final_test_acc_dict.items():
        print(f"🔹 {label}: {acc:.2f}%")

if __name__ == '__main__':
    main()
