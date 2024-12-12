import os
import dlib
import cv2
import numpy as np
import pickle
import time
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# โหลดโมเดล Dlib
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("./models/dlib_face_recognition_resnet_model_v1.dat")

def augment_image(image):
    augmented_images = []
    augmented_images.append(cv2.flip(image, 1))  # Flip แนวนอน
    augmented_images.append(cv2.GaussianBlur(image, (5, 5), 0))  # เพิ่ม blur
    return augmented_images

# ฟังก์ชันดึง face encodings
def get_face_encodings(image):
    faces = detector(image)
    if len(faces) == 0:
        return []
    encodings = []
    for face in faces:
        shape = shape_predictor(image, face)
        encoding = face_rec_model.compute_face_descriptor(image, shape)
        encodings.append(np.array(encoding))
    return encodings

# โหลดข้อมูลจากโฟลเดอร์

def load_faces_from_folder(folder_path):
    encodings, labels = [], []

    for person_name in tqdm(os.listdir(folder_path), desc="Loading People"):
        person_folder = os.path.join(folder_path, person_name)
        if os.path.isdir(person_folder):
            for file_name in os.listdir(person_folder):
                file_path = os.path.join(person_folder, file_name)
                image = cv2.imread(file_path)
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Original Image
                encoding_list = get_face_encodings(image_rgb)
                if encoding_list is not None:
                    for encoding in encoding_list:
                        encodings.append(encoding)
                        labels.append(person_name)
                
                # Augmented Images
                for aug_image in augment_image(image_rgb):
                    encoding_list = get_face_encodings(aug_image)
                    if encoding_list is not None:
                        for encoding in encoding_list:
                            encodings.append(encoding)
                            labels.append(person_name)
    return encodings, labels

# โหลดข้อมูลใบหน้า
start_time = time.time()
print("Loading data...")
dataset_path = "./Frames/"
encodings, labels = load_faces_from_folder(dataset_path)
print(f"Total faces loaded: {len(encodings)}")
print(f"Data loading completed in {time.time() - start_time:.2f} seconds")

# แบ่งข้อมูลเป็น Train และ Test Set
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(encodings, labels, test_size=0.2, random_state=42, stratify=labels)

# เทรนโมเดล SVM
start_train = time.time()
print("Training the model...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
print(f"Model training completed in {time.time() - start_train:.2f} seconds")

# บันทึกโมเดล
with open("face_recognition_model.pkl", "wb") as f:
    pickle.dump(clf, f)
print("Model trained and saved as 'face_recognition_model.pkl'.")

# ทดสอบโมเดล
print("Testing the model...")
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Script completed in {time.time() - start_time:.2f} seconds")
