import torch #работа с тензорами и нейронными сетями.
import torch.nn as nn #определение нейронных сетей и функций потерь
from torch.utils.data import DataLoader #работа с наборами данных и загрузчиками данных
from torchvision import datasets, transforms #определение преобразований, применяемых к изображениям.
import matplotlib.pyplot as plt #создание графиков и визуализация данных
import numpy as np #работа с многомерными массивами данных
import torchvision.models as models #модели для работы с изображениями.
import os

project_dir = os.path.dirname(os.path.abspath(__file__))

#путь к директории с данными
test_folder = os.path.join(project_dir, 'dataset', 'test')

#трансформация изображений
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=test_folder, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#загрузка обученной модели
checkpoint_path = 'checkpoint.pth.tar'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8) #замена 18-ого слоя на 8 классов проекта
model = model.to(device)

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

def test_model(model):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels

#проверка модели на тестовом наборе и построение графика
predictions, true_labels = test_model(model)

#расчет точности на тестовом наборе
correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
total = len(true_labels)
accuracy = correct / total * 100
print(f'Test Accuracy: {accuracy:.2f}%')

#подсчет количества правильно классифицированных изображений для каждого класса
class_correct = [0 for _ in range(8)]
class_total = [0 for _ in range(8)]

for pred, true in zip(predictions, true_labels):
    if pred == true:
        class_correct[true] += 1
    class_total[true] += 1

#получение списка имен классов из набора данных
class_names = test_dataset.classes

#построение графика с названиями классов
plt.figure(figsize=(10, 5))
plt.bar(class_names, class_correct, color='green', label='Correct')
plt.bar(class_names, np.subtract(class_total, class_correct), bottom=class_correct, color='red', label='Incorrect')
plt.title('Correct vs Incorrect Predictions by Class')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()
