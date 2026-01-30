# Brain Tumor Classification using Ensemble Learning and Grad-CAM
This repository contains code for classifying brain tumors (glioma, meningioma, pituitary, no tumor) from MRI images using an ensemble of deep learning models. It leverages pre-trained models like ResNet50, DenseNet121, and ConvNeXt-Tiny, combining their predictions for improved accuracy. Additionally, it includes implementations of Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability, visualizing which parts of the image contribute most to the classification decision.
Features
Ensemble Learning: Combines predictions from multiple state-of-the-art CNN architectures (ResNet50, DenseNet121, ConvNeXt-Tiny).
Weighted Ensemble: Utilizes custom weights for each model's contribution to the final prediction, optimized for performance.
Image Data Augmentation: Employs ImageDataGenerator for augmenting training data to improve model robustness.
Pre-trained Models: Leverages timm library for easy access to pre-trained models on ImageNet.
Grad-CAM Visualization: Implements Grad-CAM to provide visual explanations of model predictions, highlighting important regions in MRI scans.
Comprehensive Evaluation: Includes metrics such as accuracy, F1-score, and confusion matrices.
External Dataset Validation: Evaluates model performance on a separate, unseen dataset to ensure generalization.
Installation
To run this project, you'll need to set up your Python environment and install the necessary libraries. It's recommended to use a virtual environment.

Clone the repository:

git clone <repository_url>
cd brain-tumor-classification
Install dependencies:

pip install -r requirements.txt
# Or manually install:
pip install torch torchvision timm scikit-learn matplotlib seaborn pytorch-grad-cam opencv-python Pillow kaggle
Kaggle API Key (for dataset download):

Go to your Kaggle account settings and create a new API token.
Download kaggle.json.
In your Colab environment or local machine, create a .kaggle directory in your home directory (~/.kaggle/).
Copy kaggle.json to ~/.kaggle/ and set appropriate permissions:
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
Usage
This project is designed to be run in a Google Colab environment due to its GPU acceleration and ease of Kaggle dataset integration. The notebook performs the following steps:

Dataset Download: Automatically downloads the necessary brain MRI datasets from Kaggle.
Data Preprocessing: Cleans and prepares the image datasets, merging similar classes from different datasets.
Model Training: Trains an ensemble of ResNet50, DenseNet121, and ConvNeXt-Tiny models.
Evaluation: Evaluates individual and ensemble model performance using accuracy and F1-score, and visualizes results with confusion matrices and performance comparison plots.
Grad-CAM Analysis: Generates Grad-CAM visualizations for individual models and the ensemble to understand their decision-making process.
Custom Image Prediction: Allows uploading a new MRI image for prediction and Grad-CAM visualization by the trained ensemble model.
To run the notebook:

Open the .ipynb file in Google Colab.
Go to Runtime -> Run all or execute cells one by one.
Follow the prompts for uploading kaggle.json and custom images.
Acknowledgements
The datasets used in this project are from Kaggle:
Brain Tumor MRI Dataset
Brain Tumor Classification (MRI)
This work utilizes various open-source libraries, including PyTorch, timm, scikit-learn, matplotlib, and pytorch-grad-cam.
<img width="1490" height="418" alt="image" src="https://github.com/user-attachments/assets/4bb44c22-05df-4613-983a-21bc0e17a4c3" />
<img width="1490" height="418" alt="image" src="https://github.com/user-attachments/assets/3bb1e1a3-7724-427f-ba74-adb2e470b3b9" />
<img width="1490" height="418" alt="image" src="https://github.com/user-attachments/assets/3e89f947-732b-4afa-a98c-3a8eea744dcc" />
<img width="1490" height="418" alt="image" src="https://github.com/user-attachments/assets/047abc52-c8e6-4149-937e-1aba7a98b94f" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/de1fd416-b1bb-433c-b0b3-d3f51f07bf63" />
<img width="1490" height="418" alt="image" src="https://github.com/user-attachments/assets/fbbbaf8e-0e8b-4cfe-a578-2684e3ba4309" />
