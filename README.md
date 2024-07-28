# Fabric Defect Detection Project

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

This project successfully demonstrates the application of machine learning techniques for detecting defects in fabric images. By leveraging the power of pre-trained models like VGG16 for feature extraction and training various classifiers, including Support Vector Machine (SVM), Logistic Regression, and Convolutional Neural Networks (CNN), we aimed to achieve high accuracy in distinguishing between 'good' and 'oil spot' fabric images.

### Key Highlights

1. **Data Preprocessing and Feature Extraction:**
   - Utilized the VGG16 model to preprocess and extract features from fabric images.
   - Created a robust pipeline for loading, processing, and labeling the images from the dataset.

2. **Model Training:**
   - Trained SVM and Logistic Regression classifiers using the extracted features.
   - Built and trained a CNN model directly on the images for end-to-end learning.

3. **Evaluation:**
   - Implemented custom evaluation functions to assess the performance of the models on random samples of images.
   - Achieved competitive accuracy with SVM, Logistic Regression, and CNN models, demonstrating the effectiveness of the approach.

4. **Model Saving:**
   - Saved the trained models (SVM, Logistic Regression) as `.pkl` files for future use.

### Future Work

- **Hyperparameter Tuning:** Optimize hyperparameters for the SVM, Logistic Regression, and CNN models to improve their performance. This includes tuning the number of layers, units, dropout rates, and learning rates for CNN, as well as parameters like C and gamma for SVM.
- **Data Augmentation:** Introduce data augmentation techniques to increase the diversity and robustness of the training dataset.
- **Advanced Models:** Explore more advanced models like ResNet, Inception, or EfficientNet to potentially improve classification accuracy.
- **Deployment:** Develop and deploy the model in a real-world setting to evaluate its practical performance and make necessary adjustments.

This project serves as a foundational step towards creating an automated system for fabric defect detection, with the potential for significant impact in quality control processes in the textile industry. Further improvements and optimizations can enhance the accuracy and reliability of the system, making it a valuable tool for industry applications.

## Repository Contents

- **data/**: Directory containing the fabric images categorized into 'good' and 'oil spot'.
- **models/**: Directory containing the trained models saved as `.pkl` files.
- **notebooks/**: Jupyter notebooks for data preprocessing, model training, evaluation, and hyperparameter tuning.
- **scripts/**: Python scripts for feature extraction, model training, and evaluation.
- **README.md**: This readme file.
