import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import matplotlib.pyplot as plt
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms
from PIL import Image, ImageDraw
import cv2
import numpy as np

# Definir el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar modelo VGG16 con pesos preentrenados
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights)
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, 5)

# Cargar pesos entrenados por el usuario
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Cronómetro para medir tiempo
def TicTocGenerator():
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti

TicToc = TicTocGenerator()

def toc(print_time=True):
    delta = next(TicToc)
    if print_time:
        print("Tiempo transcurrido: %.4f segundos\n" % delta)

def tic():
    toc(False)

# Función para predecir clase de una imagen
def predict_image(image_path, model, device):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    return predicted.item()

# Carpeta de imágenes
folder_path = 'BD_New_DKC/testing_dkc'
subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path)
              if os.path.isdir(os.path.join(folder_path, d))]

# Dimensiones del rectángulo
rect_width = 8
rect_height = 50

# Configuración del video
output_video_path = 'output_video.avi'
fps = 10  # Frames por segundo
frame_size = None
video_writer = None

# Función para dibujar un carro de F1
def draw_f1_car(width=40, height=70, color=(255, 0, 0, 255)):
    """Dibuja una silueta simplificada de un carro de F1"""
    car_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(car_img)
    
    # Cuerpo principal del carro (más ancho en el medio)
    body_points = [
        (width//2 - 3, 5),   # Nariz
        (width//2 + 3, 5),
        (width//2 + 8, 15),  # Frente
        (width//2 + 10, 25), # Cockpit inicio
        (width//2 + 12, 45), # Parte trasera
        (width//2 + 8, 55),  # Alerón trasero
        (width//2 - 8, 55),
        (width//2 - 12, 45),
        (width//2 - 10, 25),
        (width//2 - 8, 15),
        (width//2 - 3, 5),   # Cerrar forma
    ]
    draw.polygon(body_points, fill=color)
    
    # Alerón delantero
    draw.rectangle([width//2 - 12, 10, width//2 + 12, 14], fill=color)
    
    # Alerón trasero
    draw.rectangle([width//2 - 14, 52, width//2 + 14, 58], fill=color)
    
    # Ruedas (4 círculos)
    wheel_color = (50, 50, 50, 255)  # Negro/gris oscuro
    wheel_size = 5
    # Rueda delantera izquierda
    draw.ellipse([width//2 - 16, 18, width//2 - 16 + wheel_size*2, 18 + wheel_size*2], fill=wheel_color)
    # Rueda delantera derecha
    draw.ellipse([width//2 + 16 - wheel_size*2, 18, width//2 + 16, 18 + wheel_size*2], fill=wheel_color)
    # Rueda trasera izquierda
    draw.ellipse([width//2 - 18, 42, width//2 - 18 + wheel_size*2, 42 + wheel_size*2], fill=wheel_color)
    # Rueda trasera derecha
    draw.ellipse([width//2 + 18 - wheel_size*2, 42, width//2 + 18, 42 + wheel_size*2], fill=wheel_color)
    
    # Cabina del piloto
    draw.ellipse([width//2 - 4, 28, width//2 + 4, 35], fill=(255, 255, 0, 255))  # Amarillo
    
    return car_img

# Función para verificar si el carro está en la pista
def check_car_on_road(car_x, car_y, car_width, car_height, img_width, img_height, road_margin=0.15):
    """
    Verifica si el carro está dentro de los límites de la pista.
    
    Args:
        car_x: Posición X del carro (centro)
        car_y: Posición Y del carro (centro)
        car_width: Ancho del carro
        car_height: Alto del carro
        img_width: Ancho de la imagen
        img_height: Alto de la imagen
        road_margin: Margen lateral de la pista (0.15 = 15% de cada lado es fuera de pista)
    
    Returns:
        bool: True si está en la pista, False si se salió
    """
    # Calcular bordes de la pista (dejando margen lateral)
    road_left = img_width * road_margin
    road_right = img_width * (1 - road_margin)
    road_top = 0
    road_bottom = img_height
    
    # Calcular bordes del carro
    car_left = car_x - car_width // 2
    car_right = car_x + car_width // 2
    car_top = car_y - car_height // 2
    car_bottom = car_y + car_height // 2
    
    # Verificar si alguna parte del carro está fuera de la pista
    off_road = (
        car_left < road_left or
        car_right > road_right or
        car_top < road_top or
        car_bottom > road_bottom
    )
    
    return not off_road

# Configuración del video
output_video_path = 'output_video.avi'
fps = 10  # Frames por segundo
frame_size = None
video_writer = None

# Contadores para medir rendimiento
success_count = 0  # Frames donde el carro está en la pista
fail_count = 0     # Frames donde el carro sale de la pista
total_frames = 0   # Total de frames procesados

# Iterar por carpeta
for subfolder in subfolders:
    image_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    image_files.sort()  # Opcional: ordena las imágenes

    for i in range(0, len(image_files), 1):
        image_file = image_files[i]

        # Predicción
        result = predict_image(image_file, model, device)
        print(f"Imagen: {os.path.basename(image_file)}, Prediccion: {result}")

        # Definir ángulo según clase
        angle_map = {0: 55, 1: 15, 2: 0, 3: -15, 4: -55}
        rotation_angle = angle_map.get(result, 0)

        # Abrir imagen original
        img = Image.open(image_file)
        img_width, img_height = img.size

        # Verificar si el carro está en la pista usando detección mejorada
        car_center_x = img_width // 2  # El carro está centrado horizontalmente
        car_center_y = img_height - 35  # Centro vertical del carro
        
        # Detectar bordes de la pista usando múltiples métodos
        img_cv = np.array(img)
        img_gray_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        
        # Aplicar blur para reducir ruido
        img_blur = cv2.GaussianBlur(img_gray_cv, (5, 5), 0)
        
        # Detectar bordes con Canny
        edges = cv2.Canny(img_blur, 50, 150)
        
        # Zona de interés: donde está el carro (parte inferior de la imagen)
        roi_y = int(img_height * 0.5)
        roi_height = int(img_height * 0.5)
        roi = edges[roi_y:roi_y+roi_height, :]
        roi_gray = img_gray_cv[roi_y:roi_y+roi_height, :]
        
        # Analizar múltiples filas para encontrar los bordes de la carretera
        road_left_bound = 0
        road_right_bound = img_width
        
        # Analizar desde la fila donde está el carro hacia arriba
        car_row_in_roi = roi_height - 10  # Aproximadamente donde está el carro
        num_rows_to_check = 30
        
        left_edges = []
        right_edges = []
        
        for row_idx in range(max(0, car_row_in_roi - num_rows_to_check), min(roi_height, car_row_in_roi + 5)):
            row = roi[row_idx, :]
            edge_positions = np.where(row > 0)[0]
            
            if len(edge_positions) > 0:
                # Filtrar para encontrar los bordes principales
                groups = []
                if len(edge_positions) > 0:
                    current_group = [edge_positions[0]]
                    
                    for i in range(1, len(edge_positions)):
                        if edge_positions[i] - edge_positions[i-1] < 15:
                            current_group.append(edge_positions[i])
                        else:
                            groups.append(current_group)
                            current_group = [edge_positions[i]]
                    groups.append(current_group)
                
                # Si hay al menos 2 grupos, tomar el más izquierdo y derecho
                if len(groups) >= 2:
                    left_edges.append(int(np.mean(groups[0])))
                    right_edges.append(int(np.mean(groups[-1])))
                elif len(groups) == 1:
                    # Si solo hay un grupo, podría ser un borde
                    edge_pos = int(np.mean(groups[0]))
                    if edge_pos < img_width // 2:
                        left_edges.append(edge_pos)
                    else:
                        right_edges.append(edge_pos)
        
        # Calcular promedios de los bordes detectados
        if len(left_edges) > 0:
            road_left_bound = int(np.median(left_edges))
        if len(right_edges) > 0:
            road_right_bound = int(np.median(right_edges))
        
        # Si no se detectaron bordes, usar umbrales por defecto
        if road_left_bound == 0 and road_right_bound == img_width:
            # Usar detección basada en color/intensidad
            # El asfalto suele ser más oscuro que los bordes
            threshold = 80
            dark_pixels = np.where(roi_gray < threshold)
            
            if len(dark_pixels[1]) > 0:
                road_left_bound = int(np.min(dark_pixels[1]))
                road_right_bound = int(np.max(dark_pixels[1]))
        
        # Margen de seguridad ajustable
        safety_margin = 25
        
        # Verificar si el carro está dentro de los límites detectados
        car_left = car_center_x - 30
        car_right = car_center_x + 30
        
        on_road = (
            car_left > road_left_bound + safety_margin and
            car_right < road_right_bound - safety_margin
        )
        
        # Actualizar contadores de rendimiento
        total_frames += 1
        if on_road:
            success_count += 1
        else:
            fail_count += 1
        
        # Cambiar color según si está en la pista o no
        if on_road:
            car_color = (255, 0, 0, 255)  # Rojo - En la pista
            status_text = "ON TRACK"
            status_color = (0, 255, 0)  # Verde
            warning_text = ""
        else:
            car_color = (255, 165, 0, 255)  # Naranja - Fuera de pista
            status_text = "OFF TRACK!"
            status_color = (255, 0, 0)  # Rojo
            warning_text = "⚠️ WARNING: CAR OFF ROAD!"
            print(f"*** WARNING: {os.path.basename(image_file)} - Car went OFF ROAD! ***")
        
        # Crear carro de F1
        f1_car = draw_f1_car(width=40, height=70, color=car_color)
        f1_car = f1_car.rotate(rotation_angle, expand=True, resample=Image.BICUBIC)

        # Pegar carro de F1 rotado
        paste_x = (img_width - f1_car.width) // 2
        paste_y = img_height - f1_car.height
        img.paste(f1_car, (paste_x, paste_y), f1_car)
        
        # Agregar texto de estado en la imagen
        draw_status = ImageDraw.Draw(img)
        font_size = 20
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", font_size)
            font_warning = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
            font_warning = font
        
        # Posición del texto (esquina superior izquierda)
        text_position = (10, 10)
        draw_status.text(text_position, status_text, fill=status_color, font=font)
        
        # Agregar mensaje de advertencia si está fuera de pista
        if warning_text:
            warning_position = (10, 40)
            # Fondo rojo semi-transparente para mejor visibilidad
            warning_bg = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw_bg = ImageDraw.Draw(warning_bg)
            bbox = draw_status.textbbox(warning_position, warning_text, font=font_warning)
            draw_bg.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=(255, 0, 0, 180))
            img = Image.alpha_composite(img.convert('RGBA'), warning_bg).convert('RGB')
            
            # Dibujar texto de advertencia en blanco
            draw_status = ImageDraw.Draw(img)
            draw_status.text(warning_position, warning_text, fill=(255, 255, 255), font=font_warning)

        # Agregar estadísticas en tiempo real en la imagen
        success_pct = (success_count / total_frames * 100) if total_frames > 0 else 0
        fail_pct = (fail_count / total_frames * 100) if total_frames > 0 else 0
        
        stats_text = f"SUCCESS: {success_pct:.1f}% | FAIL: {fail_pct:.1f}%"
        stats_position = (10, img_height - 30)
        stats_color = (0, 255, 0) if fail_pct < 30 else (255, 165, 0) if fail_pct < 50 else (255, 0, 0)
        draw_status.text(stats_position, stats_text, fill=stats_color, font=font)

        # Convertir PIL Image a numpy array para OpenCV
        img_array = np.array(img)
        
        # Convertir RGB a BGR para OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array

        # Inicializar el escritor de video con el tamaño del primer frame
        if video_writer is None:
            height, width = img_bgr.shape[:2]
            frame_size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
            print(f"Iniciando grabación de video: {output_video_path}")
            print(f"Tamaño del frame: {frame_size}, FPS: {fps}")

        # Escribir el frame al video
        video_writer.write(img_bgr)

        # Mostrar imagen (opcional, puedes comentar estas líneas si no quieres mostrar)
        plt.imshow(img)
        plt.title(f"Imagen: {os.path.basename(image_file)}")
        plt.axis('off')
        plt.draw()
        plt.pause(0.01)

# Liberar el escritor de video
if video_writer is not None:
    video_writer.release()
    print(f"\nVideo guardado exitosamente en: {output_video_path}")
else:
    print("No se procesaron imágenes para crear el video.")

# Calcular y mostrar estadísticas finales
print("\n" + "="*60)
print("📊 ESTADÍSTICAS FINALES DE RENDIMIENTO")
print("="*60)
print(f"Total de frames procesados: {total_frames}")
print(f"✅ SUCCESS (On Track): {success_count} frames ({success_count/total_frames*100:.2f}%)")
print(f"❌ FAIL (Off Track): {fail_count} frames ({fail_count/total_frames*100:.2f}%)")
print("="*60)

if total_frames > 0:
    success_rate = success_count / total_frames * 100
    if success_rate >= 90:
        print("🏆 EXCELLENTE! El carro mantuvo la ruta muy bien")
    elif success_rate >= 70:
        print("👍 BUENO! El carro mantuvo la ruta aceptablemente")
    elif success_rate >= 50:
        print("⚠️ REGULAR! El carro necesita mejorar el seguimiento de ruta")
    else:
        print("❌ NECESITA MEJORA! El carro se sale frecuentemente de la ruta")
print("="*60 + "\n")

plt.close('all')
