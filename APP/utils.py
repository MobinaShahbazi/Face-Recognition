from tensorflow.keras.models import load_model
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class FaceRecognition:
    def __init__(self):
        self.database = {}
        self.model = self.load_custom_model()

    def load_custom_model(self):
        """Load ResNet50 pre-trained model (without top layers) for feature extraction."""
        return load_model("resnet50_model.h5")

    def detect_faces(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces

    def preprocess_face(self, face_image, target_size=(224, 224)):
        face_image = cv2.resize(face_image, target_size)
        face_image = face_image.astype(np.float32)
        face_image[..., 0] -= 103.939
        face_image[..., 1] -= 116.779
        face_image[..., 2] -= 123.68
        face_image = np.expand_dims(face_image, axis=0)
        return face_image

    # def preprocess_face(self,face_image, target_size=(224, 224)):
    #     face_image = cv2.resize(face_image, target_size)
    #     face_image = np.expand_dims(face_image, axis=0)
    #     face_image = preprocess_input(face_image)
    #     return face_image

    def extract_features(self, face_image):
        preprocessed_image = self.preprocess_face(face_image)
        features = self.model.predict(preprocessed_image)
        return normalize(features)

    def register_face(self, name, face_image):
        features = self.extract_features(face_image)
        if name in self.database:
            self.database[name]["features"].append(features)
            self.database[name]["images"].append(face_image)
        else:
            self.database[name] = {"features": [features], "images": [face_image]}
        print(f"Face registered for {name}")

    def register_all(self, image_folder):
        for person_folder in os.listdir(image_folder):
            person_path = os.path.join(image_folder, person_folder, 'register')
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_file)
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Failed to load image: {img_path}")
                        continue
                    faces = self.detect_faces(image)
                    for (x, y, w, h) in faces:
                        face = image[y:y + h, x:x + w]
                        self.register_face(person_folder, face)

    def recognize_face(self, face_image, threshold=1):
        features = self.extract_features(face_image).flatten()
        min_avg_dissimilarity = float('inf')
        recognized_name = None

        for name, data in self.database.items():
            dissimilarities = []
            for db_feature in data["features"]:
                db_feature = db_feature.flatten()
                similarity = cosine_similarity(features.reshape(1, -1), db_feature.reshape(1, -1))[0][0]
                dissimilarity = 1 - similarity
                dissimilarities.append(dissimilarity)

            avg_dissimilarity = np.mean(dissimilarities)
            if avg_dissimilarity < min_avg_dissimilarity:
                min_avg_dissimilarity = avg_dissimilarity
                recognized_name = name

        if min_avg_dissimilarity < threshold:
            return f"Recognized as {recognized_name.replace('_', ' ')}"
        else:
            return "Face not recognized"

    def authenticate(self, test_image_path):
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print(f"Failed to load test image: {test_image_path}")
            return "Error: Image not loaded"

        faces = self.detect_faces(test_image)
        if len(faces) == 0:
            return "No face detected"

        for (x, y, w, h) in faces:
            face = test_image[y:y + h, x:x + w]
            return self.recognize_face(face)


# # Example Usage:
# registered_path = "/content/drive/My Drive/Colab Notebooks/ML/Project/final/registered"
# face_recog = FaceRecognition()
# face_recog.register_all(registered_path)
# print(face_recog.authenticate("img.jpg"))