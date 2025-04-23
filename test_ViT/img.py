import os
from PIL import Image
import random
from torchvision import transforms

# Đường dẫn gốc
base_dir = os.path.dirname(__file__)

# Đường dẫn đến thư mục chứa dữ liệu cũ và mới
input_dirs = {
    "glioma_tumor": os.path.join(base_dir, "data", "Train_old", "glioma_tumor"),
    "meningioma_tumor": os.path.join(base_dir, "data", "Train_old", "meningioma_tumor"),
    "no_tumor": os.path.join(base_dir, "data", "Train_old", "no_tumor"),
    "pituitary_tumor": os.path.join(base_dir, "data", "Train_old", "pituitary_tumor")
}

output_dirs = {
    "glioma_tumor": os.path.join(base_dir, "data", "Training", "glioma_tumor"),
    "meningioma_tumor": os.path.join(base_dir, "data", "Training", "meningioma_tumor"),
    "no_tumor": os.path.join(base_dir, "data", "Training", "no_tumor"),
    "pituitary_tumor": os.path.join(base_dir, "data", "Training", "pituitary_tumor")
}

# Mục tiêu: 10,000 ảnh mỗi nhãn
target_count = 10000

# Augmentation transforms
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomAffine(10, translate=(0.1, 0.1)),
])

# Function to apply augmentations
def augment_and_save(image, label_dir, count):
    # Áp dụng augmentation
    img_aug = augmentation_transforms(image)
    # Lưu ảnh augmented
    augmented_file = os.path.join(label_dir, f"aug_{count}.jpg")
    img_aug.save(augmented_file)

# Tạo augmented dataset
for label, input_dir in input_dirs.items():
    print(f"Processing label: {label}, input directory: {input_dir}")
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        continue

    output_dir = output_dirs[label]
    os.makedirs(output_dir, exist_ok=True)

    # Lấy danh sách ảnh gốc
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    current_count = len(image_files)

    # Copy ảnh gốc vào folder Training
    for i, img_path in enumerate(image_files):
        img = Image.open(img_path).convert("RGB")
        img.save(os.path.join(output_dir, f"original_{i}.jpg"))

    # Augment ảnh mới cho đến khi đạt target_count
    while current_count < target_count:
        img_path = random.choice(image_files)
        image = Image.open(img_path).convert("RGB")
        augment_and_save(image, output_dir, current_count)
        current_count += 1

    print(f"Total images for {label}: {current_count}")

print("Augmentation complete!")
