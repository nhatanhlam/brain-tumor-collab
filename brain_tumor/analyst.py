from fastai.vision.all import *
from fastai.vision.widgets import ImageClassifierCleaner
import shutil
import os

def clean_data():
    """Chạy ImageClassifierCleaner để lọc ảnh bị nhiễu."""
    # Load mô hình đã train
    learn = load_learner("models/best_model.pkl")
    learn.dls.cpu()  # Chuyển DataLoaders về CPU để tránh lỗi multiprocessing trên Windows

    # Kiểm tra nếu Dataloader không có dữ liệu
    if len(learn.dls.valid) == 0:
        print("⚠ Tập validation trống! Hãy kiểm tra lại dữ liệu.")
        return

    # Lấy dự đoán (fix lỗi unpacking)
    preds = learn.get_preds(dl=learn.dls.valid, with_input=False, with_decoded=True)
    if len(preds) == 3:  # Chỉ có 3 giá trị thay vì 4
        probs, targs, preds = preds  # Bỏ biến `losses`
    else:
        probs, targs, preds, _ = preds  # Giữ nguyên nếu đủ 4 giá trị

    # Khởi tạo ImageClassifierCleaner
    cleaner = ImageClassifierCleaner(learn)

    # Hiển thị giao diện lọc ảnh
    print("🚀 Mở giao diện ImageClassifierCleaner...")
    cleaner

    # Xóa ảnh bị đánh giá sai
    for idx in cleaner.delete():
        img_path = cleaner.fns[idx]
        incorrect_path = Path("data/Incorrect_Images/")
        incorrect_path.mkdir(parents=True, exist_ok=True)
        shutil.move(img_path, incorrect_path / img_path.name)
    
    # Cập nhật nhãn của ảnh
    for idx, new_label in cleaner.change():
        img_path = cleaner.fns[idx]
        correct_folder = Path("data/Training") / new_label
        correct_folder.mkdir(parents=True, exist_ok=True)
        shutil.move(img_path, correct_folder / img_path.name)

    print("✅ Dữ liệu đã được làm sạch! Những ảnh sai đã được di chuyển vào 'data/Incorrect_Images/'.")

if __name__ == '__main__':
    print("🔍 Đang khởi động ImageClassifierCleaner...")
    clean_data()
