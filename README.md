# Autism-Detection-using-machine-learning
Autism Detection Using Machine Learning
This project aims to detect autism from images using a machine learning approach. The steps involved include loading and preprocessing image data, training a LightGBM model, and evaluating its performance.

Table of Contents
->Requirements
->Dataset
->Steps
->Usage
->Results
->Acknowledgements

Requirements
1.Python 3.x
2.LightGBM
3.Scikit-learn
4.NumPy
5.Pillow (PIL)
6.Matplotlib

You can install the required packages using the following command:
"!pip install lightgbm scikit-learn numpy pillow matplotlib"

Dataset
The dataset should be organized as follows:

Copy code
AutismDataset/
    └── train/
        ├── Autistic/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── Non_Autistic/
            ├── image1.jpg
            ├── image2.jpg
            └── ...
Autistic/ folder contains images of autistic individuals.
Non_Autistic/ folder contains images of non-autistic individuals.
Steps
Load Images:

Load images from the specified folders.
Resize images to a uniform size (224x224 pixels).
Convert images to numpy arrays.
Assign labels (1 for autistic, 0 for non-autistic).
Preprocess Data:

Combine the images and labels from both classes.
Flatten the image arrays for training.
Encode the labels into numerical values.
Split Data:

Split the dataset into training and testing sets (80% training, 20% testing).
Create Datasets for LightGBM:

Create training and testing datasets using LightGBM's Dataset class.
Train the Model:

Set parameters for the LightGBM model.
Train the model using the training data.
Make Predictions:

Use the trained model to make predictions on the test set.
Classify the predictions into autistic or non-autistic.
Evaluate the Model:

Calculate the accuracy of the model.
Print the accuracy.
Visualize Some Images:

Plot some sample images from both autistic and non-autistic classes for visualization.
