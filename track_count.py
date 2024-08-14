import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

class Point:
    def __init__(self, x, y):
        # Inisialisasi titik dengan koordinat x dan y
        self.x = x
        self.y = y

    def to_tuple(self):
        # Mengubah titik menjadi tuple (x, y)
        return (self.x, self.y)

def point_in_polygon(x, y, polygon):
    # Mengecek apakah titik (x, y) berada di dalam poligon
    return cv2.pointPolygonTest(polygon, (int(x), int(y)), False) >= 0

class VehicleTracker:
    def __init__(self, model_path, polygon_points):
        # Memuat model YOLO dari file
        self.model = YOLO(model_path)
        # Mengonversi titik-titik poligon ke dalam format NumPy array
        self.polygon = np.array([point.to_tuple() for point in polygon_points], dtype=np.int32)
        # Menyimpan riwayat pelacakan kendaraan
        self.track_history = defaultdict(lambda: [])
        # Menyimpan kendaraan yang telah melewati poligon
        self.crossed_objects = {}
        # Menyimpan jumlah kendaraan berdasarkan kelas
        self.count_by_class = defaultdict(int)

    def process_frame(self, frame):
        # Menggunakan model YOLO untuk melacak kendaraan pada frame
        results = self.model.track(frame, classes=[0, 1, 2], persist=True, save=True, tracker="bytetrack.yaml")
        
        # Mengambil kotak pembatas, ID pelacakan, dan ID kelas dari hasil
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        # Menggambar hasil deteksi pada frame
        annotated_frame = results[0].plot()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
            track = self.track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            # Mengecek jika kendaraan berada dalam poligon dan belum dihitung
            if point_in_polygon(x, y, self.polygon):
                if track_id not in self.crossed_objects:
                    if len(track) > 1 and not point_in_polygon(track[-2][0], track[-2][1], self.polygon):
                        self.crossed_objects[track_id] = True
                        self.count_by_class[class_id] += 1

                # Menggambar kotak pembatas pada frame
                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

        # Menggambar poligon pada frame
        cv2.polylines(annotated_frame, [self.polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        # Menghitung jumlah total kendaraan
        total_count = sum(self.count_by_class.values())
        # Menyusun teks jumlah kendaraan berdasarkan kelas
        class_count_text = f"Bus: {self.count_by_class.get(0, 0)}, Car: {self.count_by_class.get(1, 0)}, Truck: {self.count_by_class.get(2, 0)}"
        # Menyusun teks jumlah total kendaraan
        total_count_text = f"Total: {total_count}"
        
        # Menambahkan teks pada frame
        cv2.putText(annotated_frame, class_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_frame, total_count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return annotated_frame
