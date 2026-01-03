PROJECT OVERVIEW
This project presents a machine learning–based system for automatic classification of eye diseases using retinal OCT images. The system leverages deep learning and transfer learning techniques to assist in early diagnosis of common eye diseases. A web-based interface is developed using Flask to allow users to upload images and receive real-time predictions.

The project compares the performance of multiple convolutional neural network architectures to identify the most accurate and efficient model for eye disease classification.

PROBLEM STATEMENT
Eye diseases such as Diabetic Macular Edema, Choroidal Neovascularization, and Drusen are major causes of vision loss. Traditional diagnostic techniques are expensive, time-consuming, and require skilled medical professionals. These limitations make early diagnosis difficult, especially in remote or resource-limited regions.

This project aims to provide an automated, cost-effective, and accessible solution using deep learning models trained on retinal image data.

OBJECTIVES OF THE PROJECT
The primary objectives of this project are:

To design an automated eye disease classification system using deep learning.

To apply and compare multiple transfer learning models for performance evaluation.

To preprocess and prepare retinal OCT images for model training.

To evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.

To deploy the trained models through a Flask-based web application.

DATASET
The dataset consists of retinal Optical Coherence Tomography (OCT) images categorized into four classes:

NORMAL

DRUSEN

DME (Diabetic Macular Edema)

CNV (Choroidal Neovascularization)

These grayscale images are commonly used in ophthalmology for retinal disease diagnosis.
Dataset source: Public OCT retinal dataset

MODELS USED
The following deep learning models were implemented using transfer learning:

ResNet-50

EfficientNet-B3

DenseNet121

Pretrained ImageNet weights were used to improve feature extraction and reduce training time.

TECHNOLOGY STACK
Programming Language
Python

Machine Learning Frameworks
TensorFlow
Keras

Web Framework
Flask

Libraries and Tools
NumPy
Pandas
OpenCV
Matplotlib
Seaborn

SYSTEM ARCHITECTURE
The system follows a three-layer architecture:

Data Layer
Handles image loading, preprocessing, normalization, and augmentation.

Model Layer
Contains trained deep learning models responsible for classification.

Application Layer
Flask-based web interface that allows image upload and displays prediction results with confidence scores.

RESULTS AND PERFORMANCE
All models were trained using transfer learning and evaluated on standard performance metrics.

DenseNet121 achieved the highest validation accuracy among the tested models, followed by EfficientNet-B3 and ResNet-50. The results demonstrate that deep learning models can effectively classify retinal diseases with high accuracy and reliability.

APPLICATION FEATURES

Upload retinal OCT images

Automatic disease classification

Model comparison

Real-time prediction using Flask

User-friendly interface

CONCLUSION
This project demonstrates the effectiveness of deep learning and transfer learning for eye disease classification. The system offers a scalable and affordable diagnostic support tool that can assist healthcare professionals and improve early detection of eye diseases, particularly in underserved regions.

Future enhancements may include adding more disease classes, improving model explainability, and deploying the system on cloud platforms.
