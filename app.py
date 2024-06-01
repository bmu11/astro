import os  #работа с операционной системой
import torch  #работа с тензорами и нейронными сетями
import torchvision.transforms as transforms  #преобразования изображений
from PIL import Image  #работа с изображениями
from flask import Flask, request, render_template, redirect, url_for, send_from_directory  #создание веб-приложения
from werkzeug.utils import secure_filename  #обработка имен загружаемых файлов
import torch.nn as nn  #создание и работа с нейронными сетями
from torchvision import models  #использование предобученных моделей из библиотеки torchvision

#настройка приложения Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

#создание папки для загрузок, если она не существует
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

#функция проверки допустимых форматов файлов
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

#функция загрузки модели
def load_model(checkpoint_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

#загрузка обученной модели
model = load_model('checkpoint.pth.tar')

#главная страница
@app.route('/')
def index():
    return render_template('index.html')

#обработка загрузки файлов
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            #преобразование изображения для модели
            input_image = Image.open(filepath).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(input_image).unsqueeze(0)

            #классификация изображения
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                class_names = ['asteroid', 'comet', 'galaxy', 'meteoroid', 'planet', 'satellite', 'star', 'sun']
                result = {class_names[i]: probabilities[i].item() * 100 for i in range(len(class_names))}
                final_class = class_names[probabilities.argmax().item()]

            return render_template('result.html', result=result, final_class=final_class, filename=filename)

    return render_template('index.html')

#отправка загруженного файла
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

#запуск приложения
if __name__ == '__main__':
    app.run(debug=True)
