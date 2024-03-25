import cv2

# Fungsi untuk melakukan pengenalan wajah menggunakan ekstraksi fitur dan ORB
def recognize_face(frame, face_cascade):
    # Ubah citra ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam citra
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Inisialisasi ORB
    orb = cv2.ORB_create()
    
    for (x, y, w, h) in faces:
        # Ekstraksi fitur ORB pada area wajah
        keypoints, descriptors = orb.detectAndCompute(gray[y:y+h, x:x+w], None)
        
        # Gambar kotak di sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Tambahkan label pada wajah yang terdeteksi
        
        cv2.putText(frame, 'Human', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Tampilkan hasil ekstraksi fitur ORB pada wajah
        frame_with_orb = cv2.drawKeypoints(frame, keypoints, None)
        cv2.imshow('ORB Features', frame_with_orb)
    
    return frame

# Mulai tangkap video dari webcam
cap = cv2.VideoCapture(0)

# Load Haar Cascade Classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Baca setiap frame dari video
    ret, frame = cap.read()
    
    # Lakukan pengenalan wajah menggunakan ekstraksi fitur dan ORB
    recognized_frame = recognize_face(frame, face_cascade)
    
    # Tampilkan frame hasil pengenalan wajah
    cv2.imshow('Face Recognition with ORB', recognized_frame)
    
    # Cek jika pengguna menekan tombol 'q', jika ya, hentikan program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Hentikan tangkapan video dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()