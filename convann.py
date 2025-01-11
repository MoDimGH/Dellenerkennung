import os
import xml.etree.ElementTree as ET

# Klassen-Definition
classes = ["Delle"]  # Füge hier alle deine Klassen hinzu

# Pfade
annotations_path = "./metallplatten_matt_crop/delleval"  # Pfad zu deinen XML-Dateien
output_path = "./metallplatten_matt_crop/label/delleval"      # Pfad für die YOLO-Annotationsdateien
os.makedirs(output_path, exist_ok=True)

def convert_voc_to_yolo(xml_file, output_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Bildgröße
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    yolo_annotations = []

    # Jedes Objekt in der Annotation
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in classes:
            continue  # Überspringt Klassen, die nicht in der Liste sind

        class_id = classes.index(class_name)
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # Bounding Box in YOLO-Format konvertieren
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        box_width = (xmax - xmin) / img_width
        box_height = (ymax - ymin) / img_height

        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    # Speichere die YOLO-Annotationsdatei
    base_filename = os.path.basename(xml_file).replace(".xml", ".txt")
    with open(os.path.join(output_path, base_filename), "w") as f:
        f.write("\n".join(yolo_annotations))


# Alle XML-Dateien im Ordner konvertieren
for xml_file in os.listdir(annotations_path):
    if xml_file.endswith(".xml"):
        convert_voc_to_yolo(os.path.join(annotations_path, xml_file), output_path)

print(f"Konvertierung abgeschlossen. YOLO-Annotationsdateien gespeichert in: {output_path}")
