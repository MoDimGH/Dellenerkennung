import torch
import cv2
import matplotlib.pyplot as plt

MODEL_PATH = "best.pt"
IMAGE_PATH = "test.jpeg"

print("Lade trainiertes YOLO-Modell...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
print("Modell geladen.")

image = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Führe Vorhersage durch...")
results = model(image_rgb)

detections = results.pandas().xyxy[0]
print("\nErkannte Objekte:")
print(detections)

dellen_count = len(detections)
print(f"Anzahl der erkannten Dellen: {dellen_count}")

results.show()
OUTPUT_PATH = "output_detected_dellen.jpg"

results.save()
print("Bild mit Bounding Boxes gespeichert in 'runs/detect/'.")
