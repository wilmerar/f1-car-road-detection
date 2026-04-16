import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

import os
import time

# Funciones auxiliares para medir el tiempo
def TicTocGenerator():
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti

TicToc = TicTocGenerator()

def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Tiempo transcurrido: %f segundos.\n" % tempTimeInterval)

def tic():
    toc(False)

# Dimensiones de las imágenes
Alto, Ancho = 120, 160
batch_size = 32
num_classes = 5

# Definir las transformaciones para los datos de entrenamiento y prueba
train_transforms = transforms.Compose([
    transforms.Resize((Alto, Ancho)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, shear=0.2, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((Alto, Ancho)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar los conjuntos de datos
train_dataset = datasets.ImageFolder('BD_New_DKC/training_dkc', transform=train_transforms)
test_dataset = datasets.ImageFolder('BD_New_DKC/testing_dkc', transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Cargar el modelo VGG16 preentrenado
model = models.vgg16(pretrained=True)

# Congelar todas las capas del modelo
for param in model.parameters():
    param.requires_grad = False

# Reemplazar la última capa totalmente conectada para nuestro problema de 5 clases
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, num_classes)

# Mover el modelo a la GPU si está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Definir la pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

# Configurar el aprendizaje temprano y la reducción de la tasa de aprendizaje
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=0.00001)
early_stopping_patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0

# Función de entrenamiento
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Función de validación
def validate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Entrenamiento y validación del modelo
num_epochs = 100
print("Entrenando el modelo...")
start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_model(model, test_loader, criterion, device)
    scheduler.step(val_loss)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Check early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == early_stopping_patience:
            print('Early stopping!')
            break

end_time = time.time()
print(f"Elapsed time: {end_time - start_time} seconds.")

# Función para hacer una predicción en una sola imagen
# Definir transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Función para predecir una sola imagen
def predict_image(image_path, model, device):
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Añadir una dimensión para el batch
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)

    return predicted.item()

# Prueba de predicciones
test_images = [
    ('BD_New_DKC/testing_dkc/0/izq0117.png', 0),
    ('BD_New_DKC/testing_dkc/0/izq0147.png', 0),
    ('BD_New_DKC/testing_dkc/1/semizq 0117.png', 1),
    ('BD_New_DKC/testing_dkc/1/semizq 0136.png', 1),
    ('BD_New_DKC/testing_dkc/2/adelante0081.png', 2),
    ('BD_New_DKC/testing_dkc/2/adelante0099.png', 2),
    ('BD_New_DKC/testing_dkc/3/semder0072.png', 3),
    ('BD_New_DKC/testing_dkc/3/semder0081.png', 3),
    ('BD_New_DKC/testing_dkc/4/der0187.png', 4),
    ('BD_New_DKC/testing_dkc/4/der0194.png', 4)
]

# Cargar el mejor modelo
model.load_state_dict(torch.load('best_model.pth'))

for image_path, expected in test_images:
    print(f"Probando imagen: {image_path} (Esperado: {expected})")
    tic()
    result = predict_image(image_path, model, device)
    print(f"Resultado: {result}")
    toc()
