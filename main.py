import cv2
import numpy as np
from hmmlearn import hmm
import os

# Define paths for training and testing images
training_image_paths = ['/home/sanjay17/Downloads/courier-train.png']  # Add more paths if available
test_image_path = '/home/sanjay17/Desktop/tst.png'


def check_image_path(image_path):
    if os.path.isfile(image_path):
        print(f"File exists: {image_path}")
    else:
        raise FileNotFoundError(f"File does not exist: {image_path}")


def preprocess_image(image_path):
    check_image_path(image_path)
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Unable to open image file: {image_path}")
    # Apply binarization
    _, binarized_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    return binarized_img


def extract_features(image):
    if image is None:
        raise ValueError("Extracted image is None. Cannot extract features.")
    # Flatten the image and normalize
    features = image.flatten() / 255.0
    return features


# Define an HMM model
def create_model(n_components):
    return hmm.GaussianHMM(n_components=n_components, covariance_type='diag')


def train_model(image_paths):
    X_train = []
    for path in image_paths:
        img = preprocess_image(path)
        features = extract_features(img)
        X_train.append(features)
    X_train = np.array(X_train)
    lengths = [len(features) for features in X_train]

    # Ensure there's enough data for training
    n_components = min(3, len(X_train))  # At least 3 components, or as many as data allows
    model = create_model(n_components)

    if len(X_train) > n_components:
        model.fit(X_train, lengths)
        print("Model trained successfully.")
        return model
    else:
        raise ValueError(f"Insufficient data points: {len(X_train)} for {n_components} components.")


def recognize_text(image_path, model):
    try:
        check_image_path(image_path)
        binarized_img = preprocess_image(image_path)
        features = extract_features(binarized_img)
        features = features.reshape(-1, 1)  # Reshape for HMM input

        # Predict using the HMM model
        log_likelihood = model.score(features)
        return log_likelihood
    except Exception as e:
        print(f"Error during text recognition: {e}")
        return None


# Train the model using the training image
trained_model = None
try:
    trained_model = train_model(training_image_paths)
except Exception as e:
    print(f"Error during model training: {e}")

# Recognize text in the test image
if trained_model:
    text_likelihood = recognize_text(test_image_path, trained_model)
    if text_likelihood is not None:
        print(f'Text likelihood: {text_likelihood}')
    else:
        print("Text recognition failed.")
else:
    print("Model training failed, skipping text recognition.")
