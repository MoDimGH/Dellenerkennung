import torch
import cv2
import matplotlib.pyplot as plt

# Pfad zu deinem trainierten YOLOv5-Modell
MODEL_PATH = "best.pt"  # Dein trainiertes Modell

# Bildpfad
IMAGE_PATH = "test.jpeg"

# YOLOv5-Modell laden
print("Lade trainiertes YOLO-Modell...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
print("Modell geladen.")

# Bild einlesen
image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Vorhersage
print("Führe Vorhersage durch...")
results = model(image_rgb)

# Bounding Boxes und Ergebnisse
detections = results.pandas().xyxy[0]
print("\nErkannte Objekte:")
print(detections)

# Anzahl der erkannten Dellen
dellen_count = len(detections)
print(f"Anzahl der erkannten Dellen: {dellen_count}")

# Ergebnisbild anzeigen und speichern
results.show()  # Zeigt das Bild mit Bounding Boxes
OUTPUT_PATH = "output_detected_dellen.jpg"

# Alternativer Ansatz, um das Ergebnis selbst zu speichern
results.save()  # Speichert Bilder automatisch in 'runs/detect/'
print("Bild mit Bounding Boxes gespeichert in 'runs/detect/'.")
