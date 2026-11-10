
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt
import os


# -------------------------------
# 1. Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Image Classification App",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Classification Web App")

st.write(
    "Upload an image and let our ResNet18 deep learning model "
    "predict its category."
)


# -------------------------------
# 2. Configuration
# -------------------------------

MODEL_PATH = "models/resnet18_cifar10_full.pth"

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -------------------------------
# 3. Image Transformations
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------
# 4. Load Trained Model
# -------------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(512, 10)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


# -------------------------------
# 5. Prediction Function
# -------------------------------

def predict_image(image, model):

    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    confidence, predicted = torch.max(
        probabilities,
        1
    )

    predicted_class = CLASSES[predicted.item()]

    confidence_score = confidence.item() * 100

    all_probabilities = probabilities[0].cpu().numpy()

    return (
        predicted_class,
        confidence_score,
        all_probabilities
    )


# -------------------------------
# 6. Load Model
# -------------------------------

if not os.path.exists(MODEL_PATH):

    st.error(
        "Trained model not found! "
        "Please check the models folder."
    )

    st.stop()

try:

    model = load_model()

except Exception as error:

    st.error(f"Error loading model: {error}")

    st.stop()


st.success("ResNet18 model loaded successfully!")


# -------------------------------
# 7. Image Upload Section
# -------------------------------

st.subheader("Upload Your Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Your uploaded image",
            use_container_width=True
        )

    with col2:

        st.subheader("Prediction")

        predicted_class, confidence, probabilities = (
            predict_image(image, model)
        )

        st.success(
            f"Predicted Class: {predicted_class.upper()}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)


    # -------------------------------
    # 8. Probability Chart
    # -------------------------------

    st.divider()

    st.subheader("Class-wise Prediction Probabilities")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        CLASSES,
        probabilities * 100
    )

    ax.set_xlabel("Classes")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Prediction Probability for Each Class")

    plt.xticks(rotation=45)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # -------------------------------
    # 9. Top 5 Predictions
    # -------------------------------

    st.subheader("Top 5 Predictions")

    top_indices = probabilities.argsort()[::-1][:5]

    for index in top_indices:

        class_name = CLASSES[index]

        probability = probabilities[index] * 100

        st.write(
            f"**{class_name.capitalize()}** — "
            f"{probability:.2f}%"
        )

        st.progress(float(probability / 100))


else:

    st.info(
        "Please upload an image to start classification."
    )


# -------------------------------
# 10. Footer
# -------------------------------

st.divider()

st.caption(
    "Developed using Python, PyTorch, ResNet18 and Streamlit | "
    "Dataset: CIFAR-10"
)
