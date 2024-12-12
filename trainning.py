import pickle
from sklearn.svm import SVC
import time

# Load the training data
with open("models/train_test_data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Train the model
start_train = time.time()
print("Training the model...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
print(f"Model training completed in {time.time() - start_train:.2f} seconds")

# Save the trained model
with open("models/face_recognition_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Model trained and saved as 'face_recognition_model.pkl'.")

