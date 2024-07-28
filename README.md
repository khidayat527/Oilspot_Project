# Fabric Defect Detection Project

---

## Introduction

This project aims to develop a machine learning model to detect defects in fabric images. The dataset consists of images categorized into two classes: 'good' and 'oil spot'. The goal is to leverage the power of convolutional neural networks (CNN) and classical machine learning classifiers to accurately classify the fabric images into these categories.

## Project Overview

The project consists of several main steps:

1. **PHASE 1: Data Preprocessing - Loading the dataset and load necessary imports**
   - Load images from the dataset.
   - Preprocess the images using the VGG16 model.
   - Extract features from the images for use in machine learning models.

2. **PHASE 2: Feature Extraction - Define preprocessing functions for feature extraction and image processing**
   - Use the pre-trained VGG16 model to extract features from the fabric images.

3. **PHASE 3: Model Split - Split dataset to training and validation set**
   - Split the dataset into training and test sets using `train_test_split`.

4. **PHASE 4 and 4.5: Model Training, Fitting and Testing Evaluation**
   - Train Support Vector Machine (SVM) and Logistic Regression classifiers using the extracted features.
   - Train a Convolutional Neural Network (CNN) for direct image classification.
   - Evaluate the performance of the SVM, Logistic Regression, and CNN models on a test set.
   - Implement evaluation functions to assess model performance based on all images.
   - Save to local directory the classifiers model as pkl

5. **IN-PROGRESS: Hyperparameter Tuning**
   - Optimize hyperparameters for the SVM, Logistic Regression, and CNN models to improve their performance.

## Conclusion

This project demonstrates the effectiveness of using deep learning and machine learning models to detect fabric defects. By leveraging the VGG16 model for feature extraction and employing various classifiers like SVM, Logistic Regression, and CNNs, we achieved competitive accuracy in classifying fabric images as either 'good' or 'oil spot'. The current results show promise, and further tuning and data augmentation can potentially improve the performance even more. The models and methodologies developed here can serve as a foundation for implementing automated fabric defect detection systems in the textile industry.

## Repository Contents

- `Oilspot_project.ipynb`: Jupyter notebook for data preprocessing, model training, evaluation, and hyperparameter tuning.
- `logistic_classifier.pkl`: Trained Logistic Regression model saved as a pickle file.
- `svm_classifier.pkl`: Trained SVM model saved as a pickle file.
- `README.md`: This readme file.
