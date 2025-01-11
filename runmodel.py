import torch
import cv2

# Lade dein trainiertes Modell
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/moses/Documents/Hochschule/WiSe2425/DEEPL/test/yolov5/runs/train/exp5/weights/best.pt')

# Bild einlesen und Vorhersage durchführen
image = cv2.imread('test.jpeg')
results = model(image)

# Ergebnisse anzeigen
results.show()  # Zeigt das Bild mit Bounding Boxes
