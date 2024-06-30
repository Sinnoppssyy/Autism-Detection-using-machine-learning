import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path)
            img = img.resize((224, 224))  # Resize images if needed
            img_array = np.array(img)
            images.append(img_array)  # Keep the image array in its original shape
            labels.append(label)
        except Exception as e:
            print(f"Error loading image: {img_path} - {e}")
    return np.array(images), np.array(labels)


autistic_folder = r"C:\Users\Shriram\Downloads\AutismDataset\train\Autistic"
non_autistic_folder = r"C:\Users\Shriram\Downloads\AutismDataset\train\Non_Autistic"


# Load images and labels
X_autistic, y_autistic = load_images_from_folder(autistic_folder, 1)
X_non_autistic, y_non_autistic = load_images_from_folder(non_autistic_folder, 0)


# Combine data from both classes
X = np.concatenate((X_autistic, X_non_autistic), axis=0)
y = np.concatenate((y_autistic, y_non_autistic), axis=0)


# Flatten the image arrays for training
X_flat = np.array([img.flatten() for img in X])


# Encode labels
label_encoder = preprocessing.LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)


# Create datasets for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)


# Set parameters for LightGBM
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}


# Train the model
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)


# Make predictions
y_pred_proba = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred_class = [1 if x >= 0.5 else 0 for x in y_pred_proba]


# Decode labels back to original
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred_class)


# Calculate accuracy
accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
print("Accuracy:", accuracy)


# Visualize some images
def plot_images(images, labels, title):
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title("Autistic" if labels[i] == 1 else "Non-Autistic")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


# Plot some autistic images
plot_images(X_autistic[:10], y_autistic[:10], "Sample Autistic Images")


# Plot some non-autistic images
plot_images(X_non_autistic[:10], y_non_autistic[:10], "Sample Non-Autistic Images")
