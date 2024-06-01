import os
from PIL import Image
from torchvision import transforms

train_folder = 'F://dataset/train'
augmented_train_folder = 'F://dataset/augmented_train'
test_folder = 'F://dataset/test'

#аугментирование каждого изображения по 5 раз, изменяется угол, яркость, контраст и насыщенность
os.makedirs(augmented_train_folder, exist_ok=True)

augmentations = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

def save_augmented_images(class_folder, class_name):
    class_path = os.path.join(train_folder, class_folder)
    augmented_class_path = os.path.join(augmented_train_folder, class_folder)
    os.makedirs(augmented_class_path, exist_ok=True)

    image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.png')]
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(class_path, filename)
        image = Image.open(image_path)

        for i in range(5):
            augmented_image = augmentations(image)
            augmented_image_pil = transforms.ToPILImage()(augmented_image)
            augmented_filename = f"{class_name}_{idx * 5 + i + 1}.jpg"
            augmented_image_path = os.path.join(augmented_class_path, augmented_filename)
            augmented_image_pil.save(augmented_image_path)

for class_folder in os.listdir(train_folder):
    class_path = os.path.join(train_folder, class_folder)
    if os.path.isdir(class_path):
        save_augmented_images(class_folder, class_folder)

print("Аугментация завершена и изображения сохранены.")
