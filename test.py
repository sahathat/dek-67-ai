import dlib
import cv2
import pickle
import numpy as np

# โหลดโมเดล Dlib และโมเดลที่เทรนไว้
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("./models/dlib_face_recognition_resnet_model_v1.dat")

with open("face_recognition_model.pkl", "rb") as f:
    clf = pickle.load(f)

# ฟังก์ชันดึง face encodings
def get_face_encodings(image, face):
    shape = shape_predictor(image, face)
    encoding = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(encoding)

# ฟังก์ชันวาดกรอบใบหน้า
def draw_face_box(image, face, name):
    x, y, w, h = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
    cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# ทดสอบกับภาพจากกล้อง
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(frame_rgb)
    
    for face in faces:
        encoding = get_face_encodings(frame_rgb, face)
        if encoding is not None:
            probabilities = clf.predict_proba([encoding])[0]
            max_prob = max(probabilities)
            if max_prob < 0.6:  # กำหนด threshold
                name = "Unknown"
            else:
                name = clf.classes_[np.argmax(probabilities)]
        else:
            name = "Unknown"
        
        draw_face_box(frame, face, name)
    
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
