import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import numpy as np
import glob
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from config import DATA_DIR
from pathlib import Path

# Function to load images and corresponding labels
def make_data():
    all_images = []
    category_labels = []
    color_labels = []

    files = glob.glob(str(DATA_DIR / "*/*"))

    for i, item in enumerate(files):
        try:
            # Load and resize image to 96x96
            img = Image.open(item).convert("RGB")  
            img = img.resize((96, 96))
            img = np.array(img)
            all_images.append(img)
        except Exception as e:
            print(f"[WARN] Cannot open image: {item} | {e}")
            continue

        # Extract color and category from parent folder name
        color, category = Path(item).parent.name.split("_")
        category_labels.append(category)
        color_labels.append(color)

        if i % 100 == 0:
            print(f"[INFO]: {i}/{len(files)} processed")

    category_names = sorted(list(set(category_labels)))
    color_names = sorted(list(set(color_labels)))

    all_images = np.array(all_images, dtype=np.float32) / 255.0
    category_labels = np.array(category_labels)
    color_labels = np.array(color_labels)

    return all_images, category_names, color_names, category_labels, color_labels

# Function for preprocessing data and splitting into train/test
def data_preprocessing():
    all_images, _, _, category_labels, color_labels = make_data()

    category_LB = LabelBinarizer()
    color_LB = LabelBinarizer()

    category_labels = category_LB.fit_transform(category_labels)
    color_labels = color_LB.fit_transform(color_labels)

    train_X, test_X, train_Category_Y, test_Category_Y, train_Color_Y, test_Color_Y = train_test_split(
        all_images, category_labels, color_labels, test_size=0.2, random_state=42
    )

    print("[INFO] Shapes:")
    print("train_X:", train_X.shape)
    print("train_Category_Y:", train_Category_Y.shape)
    print("train_Color_Y:", train_Color_Y.shape)

    return category_LB, color_LB, train_X, test_X, train_Category_Y, train_Color_Y, test_Category_Y, test_Color_Y

# Metric functions for evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Ensure directory exists
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Function to plot training loss curves
def plot_training_history(history, save_dir):
    ensure_dir(save_dir)

    plots = [
        ("category_output_loss", "val_category_output_loss", "Category Loss", "category_loss.png"),
        ("color_output_loss", "val_color_output_loss", "Color Loss", "color_loss.png"),
        ("loss", "val_loss", "Total Loss", "total_loss.png")
    ]

    for train_key, val_key, title, filename in plots:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[train_key], label="Train")
        plt.plot(history.history[val_key], label="Validation")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(train_key.split("_")[-1].capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

# Save evaluation metrics to table images
def save_metrics_table(y_true, y_pred, name, save_dir):
    ensure_dir(save_dir)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1-score": f1_score(y_true, y_pred, average="weighted"),
    }

    df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.scale(1, 1.5)

    plt.title(f"{name.capitalize()} Metrics")
    plt.savefig(os.path.join(save_dir, f"{name}_metrics.png"))
    plt.close()

# Save confusion matrix images
def save_confusion_matrix(y_true, y_pred, class_names, name, save_dir):
    ensure_dir(save_dir)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name.capitalize()} Confusion Matrix")

    plt.savefig(os.path.join(save_dir, f"{name}_confusion_matrix.png"))
    plt.close()