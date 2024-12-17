import pickle
from sklearn.metrics import accuracy_score, classification_report
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import cv2
import os
import dlib

# โหลดข้อมูลใบหน้า
start_time = time.time()
# Load the test data and model
with open("models/train_test_data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

with open("models/face_recognition_model.pkl", "rb") as f:
    clf = pickle.load(f)

def extract_features(image_path):
    """
    Extracts a 128-dimensional feature vector using dlib's face recognition model.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(img)
    if len(faces) == 0:
        raise ValueError(f"No faces detected in {image_path}.")
    
    # Assume the first detected face is the target (adjust if needed)
    face = faces[0]
    shape = shape_predictor(img, face)
    features = np.array(face_rec_model.compute_face_descriptor(img, shape))
    return features

# Paths
test_path = "Test\\"  # Path to test directory
shape_predictor_path = "./models/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./models/dlib_face_recognition_resnet_model_v1.dat"

# Load dlib models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Initialize variables
folder_accuracies = {}
true_labels = []
predicted_labels = []

# Initialize a list to store image paths with detection issues
no_face_detected = []
# Loop through each subfolder in the test path
for class_name in os.listdir(test_path):
    class_path = os.path.join(test_path, class_name)
    if not os.path.isdir(class_path):
        continue  # Skip non-directory items

    true_class_labels = []
    predicted_class_labels = []

    # Process each image in the subfolder
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        try:
            # Extract features
            features = extract_features(img_path)

            # Predict using the trained model
            prediction = clf.predict([features])[0]

            true_class_labels.append(class_name)
            predicted_class_labels.append(prediction)

            true_labels.append(class_name)
            predicted_labels.append(prediction)
        except Exception as e:
            if "No faces detected" in str(e):
                no_face_detected.append(f"class: {class_name} path: {img_path}")
            print(f"Error processing {img_path}: {e}")

    # Calculate accuracy for this folder
    accuracy = accuracy_score(true_class_labels, predicted_class_labels)
    folder_accuracies[class_name] = accuracy

# Bar chart visualization
plt.figure(figsize=(10, 6))
plt.bar(folder_accuracies.keys(), folder_accuracies.values())
plt.title("Accuracy by Folder")
plt.xlabel("Class/Folder Name")
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # Accuracy range between 0 and 1
plt.xticks(rotation=45, ha="right")

# Add accuracy labels on the bars
for i, (folder, acc) in enumerate(folder_accuracies.items()):
    plt.text(i, acc, f"{acc*100:.2f}", ha="center", va="bottom")

# Show chart
plt.tight_layout()

# Save the bar chart
chart_path = "./reports/test/accuracy_per_person_test.png"
plt.tight_layout()
plt.savefig(chart_path)

print(f"Bar chart saved at: {chart_path}")

# Print overall accuracy
overall_accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Overall Accuracy: {overall_accuracy:.2f}")

error_file_path = "./reports/test/no_face_detect_list.txt"

# Save the list of problematic image paths to a file
description = """ไฟล์นี้ประกอบด้วยรายการของภาพที่ไม่สามารถตรวจจับใบหน้าได้ในระหว่างกระบวนการจำแนกใบหน้า
ไฟล์เหล่านี้อาจต้องมีการตรวจสอบเพิ่มเติมหรือปรับปรุงเพื่อลดปัญหาในการตรวจจับใบหน้า"""
with open(error_file_path, "w") as f:
    # Write the description
    f.write(description + "\n\n")
    for path in no_face_detected:
        f.write(f"{path}\n")

print(f"Test image accuracy report completed in {time.time() - start_time:.2f} seconds")