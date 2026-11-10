# Image Classification Web App

A deep learning-based image classification application built using **PyTorch, ResNet18, and Streamlit**. The application allows users to upload an image and predicts its category among the 10 CIFAR-10 classes.

## Project Overview

- **Model:** ResNet18
- **Dataset:** CIFAR-10
- **Training images:** 50,000
- **Testing images:** 10,000
- **Image size:** 64 × 64
- **Training epochs:** 3
- **Test accuracy:** 75.59%
- **Interface:** Streamlit

## CIFAR-10 Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- Streamlit
- Matplotlib
- Scikit-learn

## Project Structure

```text
Image Classification App/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── resnet18_cifar10_full.pth
│
├── confusion_matrix_full.png
└── data/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd "Image Classification App"
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment (Windows)

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Model Performance

The ResNet18 model was trained for three epochs on CIFAR-10.

**Reported test accuracy: 75.59%**

The model predicts one of the ten CIFAR-10 categories for an uploaded image.

## Limitations

- The model supports only the ten CIFAR-10 categories.
- Images outside these categories may be classified incorrectly.
- Performance on real-world photographs may differ from CIFAR-10 test performance.

## Author

Developed as a deep learning and computer vision project using Python and PyTorch.