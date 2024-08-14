import cv2
from track_count import Point, VehicleTracker

# Membuka file video untuk dibaca frame-nya
cap = cv2.VideoCapture("data/toll_gate.mp4")

# Mengambil informasi dari video seperti FPS, lebar, dan tinggi
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video width: {width}, height: {height}")

# Mendefinisikan titik-titik untuk poligon area penghitung kendaraan
polygon_points = [
    Point(50, 100), 
    Point(150, 250), 
    Point(150, 250), 
    Point(550, 300)
]

# Inisialisasi objek VehicleTracker dengan jalur model deteksi dan titik-titik poligon
tracker = VehicleTracker(model_path='models/best.pt', polygon_points=polygon_points)

# Menyiapkan VideoWriter untuk menyimpan video keluaran dengan nama 'output_polygonal.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video/output_polygonal.mp4', fourcc, fps, (width, height))

# Memproses setiap frame dari video
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Mengolah frame dengan metode process_frame dari VehicleTracker
        annotated_frame = tracker.process_frame(frame)

        # Menampilkan frame dengan anotasi pada jendela
        cv2.imshow('Annotated Frame', annotated_frame)

        # Menulis frame dengan anotasi ke video keluaran
        out.write(annotated_frame)

        # Keluar dari loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Melepaskan video capture dan writer, serta menutup semua jendela OpenCV
cap.release()
out.release()
cv2.destroyAllWindows()
