# Sistem Penghitung Kendaraan

Proyek ini mengimplementasikan sistem penghitung kendaraan menggunakan deep learning, khususnya YOLOv8 untuk deteksi kendaraan dan ByteTrack untuk pelacakan kendaraan. Sistem ini memproses video input, mendeteksi dan melacak kendaraan, menghitung kendaraan yang melewati wilayah poligonal yang ditentukan, dan menghasilkan video output dengan visualisasi.

## Persyaratan

- Python 3.x
- OpenCV
- NumPy
- Ultralytics YOLOv8

## Instalasi

1. Clone repository:
    ```bash
    git clone https://github.com/YUDHA51/knowledge-tes-msib7-widya-robotics.git
    cd vehicle-counting-system
    ```

2. Buat lingkungan virtual dan aktifkan:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows gunakan `venv\Scripts\activate`
    ```

3. Instal dependensi yang diperlukan:
    ```bash
    pip install -r requirements.txt
    ```

## Persiapan Dataset

1. Unduh video untuk pengujian.
2. Konversi video menjadi frame (gambar).
3. Anotasi gambar menggunakan alat seperti Roboflow.
link:https://app.roboflow.com/project-kd8ox/tescardetection/3
4. Ekspor dataset yang telah dianotasi dan latih model YOLOv8 di Google Colab.
link:https://colab.research.google.com/drive/1L1fif8c9pujwH3dnWOaASyIzqFDw8D-W?usp=sharing 
5. Unduh model yang telah dilatih sebagai `best.pt`.

## Struktur Proyek
vehicle-counting-system/
│
├── data/
│ └── toll_gate.mp4 # File video input
│
├── models/
│ └── best.pt # Model YOLOv8 yang telah dilatih
│
├── output_video/
│ └── output_polygonal.mp4 # File video output
│
├── main.py # Script utama untuk menjalankan sistem
├── track_count.py # Script yang berisi logika pelacakan dan penghitungan
├── requirements.txt # Daftar dependensi
└── README.md # Dokumentasi proyek


## Menjalankan Sistem

1. Pastikan video input (`toll_gate.mp4`) ditempatkan di folder `data/`.
2. Jalankan script utama:
    ```bash
    python main.py
    ```

## Penjelasan Kode

### main.py

Script ini mengatur pengambilan video, menginisialisasi model YOLOv8, dan memproses setiap frame untuk mendeteksi, melacak, dan menghitung kendaraan. Hasilnya divisualisasikan dan disimpan sebagai video output.

### track_count.py

Script ini mendefinisikan kelas `Point` dan kelas `VehicleTracker`, yang berisi metode untuk melacak dan menghitung kendaraan menggunakan wilayah poligonal yang ditentukan.

## Output

Video output dengan visualisasi disimpan di folder `output_video/` sebagai `output_polygonal.mp4`.

## Informasi Tambahan

- Model mendeteksi tiga kelas: mobil, bus, dan truk.
- Kendaraan dihitung saat melewati wilayah poligonal yang ditentukan dalam frame.
- Sistem menetapkan ID unik untuk kendaraan yang dilacak dan menjaga hitungan setiap jenis kendaraan serta total hitungan.

## Pengakuan

- Proyek ini menggunakan model YOLOv8 dari Ultralytics.
- ByteTrack digunakan untuk pelacakan kendaraan.
- Dataset dibuat menggunakan Roboflow untuk anotasi.

## Profil
213030503101_YUDHA RINSAGHI UNIVERSITAS PALANGKA RAYA
14 Agustus 2024

