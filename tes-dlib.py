import dlib
import cv2
import numpy as np
import os
import pickle
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def run_comparison_from_pkl(pkl_path="data.pkl"):
    """Menjalankan verifikasi wajah menggunakan dataset dari file .pkl"""
    
    if not os.path.exists(pkl_path):
        print(f"[ERROR] File {pkl_path} tidak ditemukan!")
        return

    print(f"[INFO] Memuat dataset dari {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    cap = cv2.VideoCapture(0)
    print("[INFO] Memulai kamera untuk pengujian perbandingan metode...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Mulai hitung waktu respon (Waktu Deteksi Wajah) [cite: 533, 762]
        t_start = time.time()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = detector(rgb, 1)
        
        for rect in rects:
            # Tahap ekstraksi fitur wajah
            shape = predictor(rgb, rect)
            face_descriptor = facerec.compute_face_descriptor(rgb, shape)
            
            face_enc = np.array(face_descriptor)
            distances = np.linalg.norm(data["embeddings"] - face_enc, axis=1)
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            name = "Unknown"
            if min_dist < 0.6:
                name = data["names"][min_idx]
            
            left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} ({min_dist:.2f})", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Hitung rata-rata waktu respon deteksi wajah [cite: 763, 803]
        resp_time = time.time() - t_start
        cv2.putText(frame, f"Resp Time: {resp_time:.3f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Dlib Comparison (ResNet-34)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ganti "data.pkl" dengan path file Anda jika berbeda
    run_comparison_from_pkl("data.pkl")