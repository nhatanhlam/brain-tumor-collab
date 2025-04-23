from fastai.vision.all import *
import timm
import torch
from pathlib import Path

def safe_save_model(learner, filename):
    """Lưu mô hình an toàn để tránh lỗi CUDA khi pickling."""
    device = next(learner.model.parameters()).device
    learner.model.cpu()  # Chuyển về CPU trước khi lưu
    learner.export(f"models/{filename}.pkl")  # Lưu mô hình dưới dạng .pkl
    learner.model.to(device)  # Chuyển lại về thiết bị ban đầu

def main():
    Path('models').mkdir(exist_ok=True)

    # Định nghĩa đường dẫn dữ liệu
    data_path = Path('data/Training')
    
    # Kiểm tra xem các thư mục dữ liệu có tồn tại không
    assert all((data_path/cls).exists() for cls in ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']), \
           "Thiếu một hoặc nhiều thư mục dữ liệu!"

    # Tạo DataBlock
    brain_tumor = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(mult=1.0)
    )

    # Tạo DataLoaders
    dls = brain_tumor.dataloaders(data_path, bs=32, num_workers=0)

    # Khởi tạo mô hình ResNet101
    learn = vision_learner(dls, 'resnet101', metrics=accuracy) 
    #update resnet50 lên resnet101 tăng độ nhận diện ảnh
    learn.to_fp16()  # Sử dụng mixed precision để tăng tốc
    learn.model_dir = Path('models')

    # Biến để theo dõi mô hình tốt nhất
    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 3
    patience_counter = 0
    epoch = 0
    max_epochs = 100

    try:
        while epoch < max_epochs:
            epoch += 1
            print(f"Epoch {epoch} bắt đầu...")

            learn.fit_one_cycle(1, 3e-3)

            current_val_loss = learn.recorder.values[-1][0]
            current_val_acc = learn.recorder.values[-1][1]
            print(f"Epoch {epoch} - Loss: {current_val_loss:.4f}, Accuracy: {current_val_acc:.4f}")

            if current_val_loss < best_val_loss or current_val_acc > best_val_acc:
                best_val_loss = min(best_val_loss, current_val_loss)
                best_val_acc = max(best_val_acc, current_val_acc)
                print("Lưu mô hình tốt nhất...")
                safe_save_model(learn, 'best_model')
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Dừng training do không cải thiện liên tiếp.")
                break

    except Exception as e:
        print(f"Lỗi xảy ra: {e}")
        safe_save_model(learn, 'last_model_on_exception')
        raise

    finally:
        print("Lưu mô hình cuối cùng...")
        safe_save_model(learn, 'last_model')

if __name__ == '__main__':
    main()
