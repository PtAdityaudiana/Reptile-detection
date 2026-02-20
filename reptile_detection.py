from ultralytics import YOLO
import cv2
import os

menu = input("Pilih mode deteksi: (1) Kamera, (2) Gambar: ")

model = YOLO('best.pt')

if menu == '1':
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Kamera tidak dapat diakses.")
    else:
        print("Tekan 'q' untuk keluar.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.4, imgsz=640)

            annotated_frame = results[0].plot()
            cv2.imshow("Reptile Detection - Kamera", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

elif menu == '2':
    image_path = r"test_gambar\gambar_kadal1.jpg" #file gambar yang ingin dideteksi

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' tidak ditemukan.")
    else:
        results = model(image_path, conf=0.4)

        for r in results:
            img = r.plot()
            cv2.imshow("Detection Result - Gambar", img)
            print("Tekan tombol apapun pada jendela gambar untuk menutup.")
            cv2.waitKey(0)

        cv2.destroyAllWindows()
    
else:
    print("Pilihan tidak valid. Silakan pilih 1 atau 2.")