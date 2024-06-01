import torch #работа с тензорами
import torch.nn as nn #создание и обучение нейронных сетей
import torch.optim as optim #реализация различных алгоритмов оптимизации
from torch.utils.data import DataLoader #работа с загрузчиками данных (DataLoader)
from torchvision import datasets, transforms, models #работа с изображениями и предобученными моделями
import os #работа с операционной системой


project_dir = os.path.dirname(os.path.abspath(__file__))

#путь к директориям с данными
train_folder = os.path.join(project_dir, 'dataset', 'augmented_train')
test_folder = os.path.join(project_dir, 'dataset', 'test')

#трансформация изображений
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=train_folder, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_folder, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=False)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8) #замена 18-ого слоя на 8 кллассов проекта

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#сохранение и загрузка контрольных точек
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(filename='checkpoint.pth.tar'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0

#функция для тренировки модели
def train_model(model, criterion, optimizer, num_epochs=25, start_epoch=0):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })

    return model

def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

#загрузка контрольной точки, если она существует
#start_epoch = load_checkpoint()

#тренировка модели
#trained_model = train_model(model, criterion, optimizer, num_epochs=25, start_epoch=start_epoch)

#evaluate_model(trained_model)

def get_model(m_path, vis_model=False):
    resnet18 = models.resnet18()
    if vis_model:
        from torchsummary import summary
        summary(resnet18, input_size=(3, 224, 224), device="cpu")

    return resnet18

model_path = "./checkpoint.pth.tar"
get_model(model_path, True)