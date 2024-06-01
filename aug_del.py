import os

augmented_folder = 'F://dataset/augmented_train'

#удаление каждого 3-его и 4-ого аугментированного изображения
for class_folder in os.listdir(augmented_folder):
    class_path = os.path.join(augmented_folder, class_folder)
    if os.path.isdir(class_path):
        files = sorted(os.listdir(class_path))
        files_to_delete = [files[i] for i in range(3, len(files), 5)] + [files[i] for i in range(4, len(files), 5)]
        for file_name in files_to_delete:
            file_path = os.path.join(class_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f'Deleted: {file_path}')
