# LSML2
## ▎Overview
This project focuses on computer vision by leveraging the FOOD-101 (https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset. The dataset contains 101 classes of food images with corresponding annotations for training and testing. The goal is to develop a model capable of identifying food categories in images.

---

## ▎Task Definition

Input:  
An image of food (e.g., JPEG/PNG format).  

Output:  
The predicted food class label (from one of the 15 available classes).  

Approach:  
The solution involves training a Convolutional Neural Network (CNN) based on a pretrained ResNet-34 architecture from PyTorch's model zoo. The model is fine-tuned using transfer learning to recognize food categories from the given dataset.  

Dataset:  
The project uses the FOOD-101 dataset containing:
- Images Folder: Images organized by food class.  
- Meta Folder: Contains training and testing annotations in both JSON and TXT formats (lists of image paths grouped by class).  

---

## ▎Workflow

1. Data Preparation:
   - Extract and preprocess the dataset: Resize images to (128x128) and normalize pixel values.
   - Use annotations from the meta folder to split the data into training, validation, and testing sets.  

2. Model Design:
   - Use a pretrained ResNet-34 model.
   - Replace the final fully connected layer to output 101 categories.

3. Training:
   - Train the model using cross-entropy loss on the training set.
   - Validate the performance using a separate validation set to avoid overfitting.  

4. Evaluation:
   - Test the model's performance on the provided test set and assess final metrics like accuracy.

5. Tracking:
   - Use Weights & Biases (WandB) for logging training metrics (NOT REALIZED), validation performance, and saving the best models.

---

## ▎Model Details

- Architecture: ResNet-34 (pretrained on ImageNet).
- Loss Function: Cross-Entropy Loss.
- Optimizer: Adam with a learning rate of 0.001.
- Input Shape: (128, 128, 3).  
- Output Layer: Linear layer mapping 512 features to 101 classes.  
