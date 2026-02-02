# Brain Tumor MRI Analysis using Deep Learning

## 📌 Project Overview
This project focuses on **brain tumor detection from MRI images** using **Deep Learning (Convolutional Neural Networks – CNNs)**.  
The notebook walks through data loading, preprocessing, model building, training, and evaluation to classify MRI images into:

- **Tumor present (Yes)**
- **No tumor (No)**

The goal is to assist in automated medical image analysis by leveraging computer vision and neural networks.

---

## 📂 Dataset Structure
The dataset is expected to be organized as follows:

```
brain_tumor_dataset/
│
├── yes/        # MRI images with brain tumor
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
└── no/         # MRI images without brain tumor
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

Update the dataset path in the notebook if required:

```python
No_Data_Path = Path("Downloads/brain_tumor_dataset/no")
yes_Data_Path = Path("Downloads/brain_tumor_dataset/yes")
```

---

## 🛠️ Technologies & Libraries Used
- Python 3.x  
- NumPy & Pandas – data handling  
- Matplotlib & Seaborn – visualization  
- OpenCV (cv2) – image processing  
- TensorFlow / Keras – deep learning  
- Scikit-learn – evaluation metrics  
- PIL – image handling  

---

## 🔄 Workflow

### 1️⃣ Import Libraries & Ignore Warnings
Essential libraries are imported and unnecessary warnings are suppressed for clean output.

### 2️⃣ Load Image Paths
MRI images are loaded from `yes` and `no` folders using `Path` and `glob`.

### 3️⃣ Image Preprocessing
- Image resizing  
- Normalization  
- Label encoding  
- Train-test split  

### 4️⃣ Data Augmentation
`ImageDataGenerator` is used to improve model generalization by applying transformations such as:
- Rotation  
- Zoom  
- Flip  

### 5️⃣ Model Architecture
A **CNN model** is built using:
- Convolutional layers (`Conv2D`)  
- Max pooling layers  
- Batch normalization  
- Dropout (to reduce overfitting)  
- Dense layers for classification  

### 6️⃣ Model Training
- Optimizer: Adam / RMSprop  
- Loss function: Binary / Categorical Crossentropy  
- Evaluation using validation data  

### 7️⃣ Model Evaluation
The model is evaluated using:
- Accuracy  
- Confusion Matrix  
- Classification Report  
- ROC Curve & AUC Score  

---

## 📊 Evaluation Metrics
- Accuracy Score  
- Confusion Matrix  
- Precision, Recall, F1-score  
- ROC-AUC Curve  

These metrics help understand model performance beyond accuracy.

---

## ▶️ How to Run the Notebook

1. Clone or download the repository  
2. Install required libraries:

```bash
pip install numpy pandas matplotlib seaborn opencv-python tensorflow keras scikit-learn pillow
```

3. Place the dataset in the correct directory  
4. Open and run `Brain Analysis.ipynb` step by step  

---

## 🚀 Future Improvements
- Transfer Learning (VGG16, ResNet, MobileNet)  
- Hyperparameter tuning  
- Multi-class tumor classification  
- Deploy as a web app (Flask / Streamlit)  

---

## ⚠️ Disclaimer
This project is for **educational and research purposes only** and should not be used for real medical diagnosis without professional validation.

---

## 👩‍💻 Author
**Bamandla Gouthami**
