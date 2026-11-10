
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 2. CONFIGURATION
# ============================================================

BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 0.001
IMAGE_SIZE = 64

classes = [
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


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. LOAD CIFAR-10 DATASETS
# ============================================================

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ============================================================
# 5. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nTraining images:", len(train_dataset))
print("Testing images:", len(test_dataset))

print("\nImages used for training:", len(train_dataset))
print("Images used for testing:", len(test_dataset))


# ============================================================
# 6. LOAD PRETRAINED RESNET18
# ============================================================

print("\nLoading pretrained ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# ============================================================
# 7. FREEZE PRETRAINED LAYERS
# ============================================================

for param in model.parameters():
    param.requires_grad = False


# ============================================================
# 8. REPLACE FINAL CLASSIFICATION LAYER
# ============================================================

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    10
)

model = model.to(device)

print("\nModel:")
print(model.fc)


# ============================================================
# 9. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 10. OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 11. TRAINING
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")

for epoch in range(NUM_EPOCHS):

    # Keep pretrained backbone in evaluation mode.
    # Only train the final classification layer.

    model.eval()
    model.fc.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Calculate statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Training Accuracy: {epoch_accuracy:.2f}%"
    )


print("\nTraining completed!")


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

print("\n========================================")
print("EVALUATING MODEL")
print("========================================")

model.eval()

correct = 0
total = 0

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predicted.cpu().tolist())


# ============================================================
# 13. TEST ACCURACY
# ============================================================

test_accuracy = 100 * correct / total

print("\nFinal Test Accuracy: {:.2f}%".format(test_accuracy))


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(10)),
    target_names=classes,
    zero_division=0
)

print(report)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(10))
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ============================================================
# 16. PLOT CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(cm, cmap="Blues")

plt.title("CIFAR-10 Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    range(len(classes)),
    classes,
    rotation=45
)

plt.yticks(
    range(len(classes)),
    classes
)

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "confusion_matrix_full.png",
    dpi=300
)

plt.close()

print("\nConfusion matrix saved successfully!")


# ============================================================
# 17. SAVE MODEL CHECKPOINT
# ============================================================

os.makedirs("models", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "test_accuracy": test_accuracy
    },
    "models/resnet18_cifar10_full.pth"
)

print("\nModel saved successfully!")

print("Saved file: models/resnet18_cifar10_full.pth")


# ============================================================
# 18. PROJECT COMPLETION
# ============================================================

print("\n========================================")
print("PROJECT PIPELINE COMPLETED")
print("========================================")
