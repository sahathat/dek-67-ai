import os
import dlib
import cv2
import numpy as np

# Load pre-trained models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("./models/dlib_face_recognition_resnet_model_v1.dat")

# Function to compute face encodings
def get_face_encodings(image):
    faces = detector(image)
    if len(faces) == 0:
        return None
    shape = shape_predictor(image, faces[0])
    encoding = face_rec_model.compute_face_descriptor(image, shape)
    return np.array(encoding)

# Load known faces from folders
def load_faces_from_folder(folder_path):
    known_encodings = []
    known_names = []

    for person_name in os.listdir(folder_path):     
        person_folder = os.path.join(folder_path, person_name)
        if os.path.isdir(person_folder):
            for file_name in os.listdir(person_folder):
                file_path = os.path.join(person_folder, file_name)
                # Load the image
                image = cv2.imread(file_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Get face encoding
                encoding = get_face_encodings(image_rgb)
                if encoding is not None:
                    known_encodings.append(encoding)
                    known_names.append(person_name)  # Use folder name as the label
                    print(f"Loaded {file_name} for {person_name}")
    return known_encodings, known_names

# Path to the faces folder
faces_folder = "./faces"

# Load known faces
print("Loading known faces...")
known_encodings, known_names = load_faces_from_folder(faces_folder)
print("Finished loading faces.")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

print("Starting webcam for real-time recognition...")
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the current frame
    faces = detector(rgb_frame)
    for face in faces:
        # Compute face encoding for the current face
        encoding = get_face_encodings(rgb_frame)
        
        # Compare with known encodings (only if encodings exist)
        if encoding is not None and len(known_encodings) > 0:
            distances = np.linalg.norm(known_encodings - encoding, axis=1)
            min_distance_index = np.argmin(distances)
            
            if distances[min_distance_index] < 0.6:
                name = known_names[min_distance_index]  # Get the name from known_names
                label = f"{name}"
                color = (0, 255, 0)  # Green for match
            else:
                label = "Unknown"
                color = (0, 0, 255)  # Red for no match
        else:
            label = "Unknown"  # Default to unknown if no encodings are loaded
            color = (0, 0, 255)


        # Draw a rectangle and label
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the frame
    cv2.imshow("Face Recognition", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
