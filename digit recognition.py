"""
Handwritten Digit Recognition using MNIST dataset with SVM classifier.

This Python script loads the MNIST dataset, trains a Support Vector Machine (SVM) classifier,
evaluates the model accuracy, and allows the user to input a digit image from a file for prediction.

Requirements:
- Python 3.x
- scikit-learn
- matplotlib
- numpy

You can install the required packages via:
pip install numpy matplotlib scikit-learn

Usage:
- Run the script. It will train the model and show accuracy.
- Follow the prompt to test digit recognition on sample images.

Author: Your Name
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

def load_data():
    # Load the digits dataset from sklearn
    digits = datasets.load_digits()
    return digits

def preprocess_data(digits):
    # Flatten images to 1D vector as SVM needs samples as vectors
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data

def train_model(data, target):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.5, shuffle=False
    )
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)
    # Train the classifier
    classifier.fit(X_train, y_train)
    return classifier, X_test, y_test

def evaluate_model(classifier, X_test, y_test):
    # Predict the value of the test data
    predicted = classifier.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_test, predicted)))
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    plt.show()

def predict_sample(classifier):
    print("\nYou can test digit recognition on your own digit image.")
    print(
        "Note: Image should be 8x8 pixels grayscale (like in the MNIST dataset) and saved as a .npy file."
    )
    print("You can create test .npy files with numpy, or use this script's sample output by saving digits.images[0] for example.")

    while True:
        file_path = input("\nEnter path to .npy file with 8x8 digit image (or 'exit' to quit): ")
        if file_path.lower() == 'exit':
            print("Exiting prediction.")
            break
        try:
            sample_image = np.load(file_path)
            if sample_image.shape != (8,8):
                print("Error: Image shape must be 8x8 pixels. Your image shape:", sample_image.shape)
                continue
            # Flatten and predict
            sample_flat = sample_image.reshape(1, -1)
            prediction = classifier.predict(sample_flat)
            print("Predicted digit:", prediction[0])
            plt.imshow(sample_image, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.title(f"Predicted: {prediction[0]}")
            plt.show()
        except Exception as e:
            print(f"Error loading or processing file: {e}")

def main():
    digits = load_data()
    data = preprocess_data(digits)
    classifier, X_test, y_test = train_model(data, digits.target)
    print("Model trained successfully.")
    evaluate_model(classifier, X_test, y_test)

    predict_sample(classifier)

if __name__ == "__main__":
    main()
</content>
</create_file>