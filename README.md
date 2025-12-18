# Multi-Task Fashion Classification with Keras Functional API

This project presents a **multi-task convolutional neural network (CNN)** that simultaneously predicts the **category** and **color** of clothing images. The model is implemented using the **Keras Functional API**, enabling a flexible architecture with shared feature extraction and task-specific output branches. Separate loss functions, configurable loss weights, and structured evaluation metrics ensure robust and interpretable multi-task learning.

---

## Table of Contents

- [Dataset](#dataset)  
- [Project Overview](#project-overview)  
- [Keras Functional API](#keras-functional-api)  
- [Label Encoding](#label-encoding)  
- [Data Splitting](#data-splitting)  
- [Multi-Output Network](#multi-output-network)  
- [Loss Functions & Weighting](#loss-functions--weighting)  
- [Training](#training)  
- [Evaluation & Visualization](#evaluation--visualization)  
- [Results](#results)  
- [Repository](#repository)  

---

## Dataset

The dataset consists of clothing images annotated with both **category** and **color** labels.  
All images are resized to **96×96 pixels** and normalized to the range `[0,1]`.

| Category & Color | Number of Images |
|------------------|------------------|
| black_jeans      | 344              |
| black_shirt      | 358              |
| blue_dress       | 386              |
| blue_jeans       | 356              |
| blue_shirt       | 369              |
| red_dress        | 380              |
| red_shirt        | 332              |

- **Split**: 80% training, 20% testing  
- Each image retains both labels to ensure alignment across tasks  

> **Note**: Dataset is custom-built, organized as `color_category/imagename.jpg`.

---

## Project Overview

- **Goal**: Predict clothing **category** (e.g., jeans, dress, shirt) and **color** (e.g., red, blue, black) at the same time.  
- **Method**: Multi-task CNN with shared backbone and task-specific heads.  
- **Benefits**:
  - Shared feature extraction reduces redundancy.  
  - Independent branches allow specialized learning.  
  - Separate losses improve optimization and interpretability.  

---

## Keras Functional API

The **Functional API** supports:

- Multiple inputs and outputs  
- Shared layers across tasks  
- Flexible non-linear computation graphs  

In this project, a **shared input layer** feeds into two distinct branches:

```python
net = models.Model(
    inputs=input_layers,
    outputs=[cat_net, col_net],
    name="fashionNet"
)
```

This design enables efficient feature sharing while preserving task-specific learning.

---

## Label Encoding

Labels are converted to one-hot vectors using `LabelBinarizer` from `scikit-learn`.

**Advantages**:
- Direct support for string labels  
- Compatible with `categorical_crossentropy`  
- Simple and reliable  

**Alternatives**:

| Method         | Notes                                                                      |
|----------------|----------------------------------------------------------------------------|
| LabelEncoder   | Produces integers; requires conversion for categorical cross-entropy       |
| OneHotEncoder  | Flexible for pipelines but more complex                                    |
| to_categorical | Requires integer labels; suitable when labels are numeric                  |

---

## Data Splitting

Data is split with `train_test_split`:

```python
train_test_split(
    all_images,
    category_labels,
    color_labels,
    test_size=0.2,
    random_state=42
)
```

- Ensures 80/20 split  
- Maintains consistent pairing of category and color labels  

---

## Multi-Output Network

The network produces two outputs from a single image input:

- `category_output`: clothing category  
- `color_output`: clothing color  

Early layers are shared, while each branch has its own convolutional and dense layers.  
This improves efficiency and encourages knowledge transfer.

---

## Loss Functions & Weighting

Each output has its own loss:

```python
losses = {
    "category_output": "categorical_crossentropy",
    "color_output": "categorical_crossentropy"
}
```

**Loss weights** balance contributions:

```python
loss_weights = {"category_output": 1.0, "color_output": 1.0}
```

Weighted total loss:

```
L_total = L_category + λ * L_color
```

Strategies for tuning λ:
- Manual grid search  
- Normalization by initial loss magnitudes  
- Dynamic methods (GradNorm, uncertainty weighting, DWA)  

---

## Training

Training with `model.fit`:

```python
history = model.fit(
    x=train_X,
    y={"category_output": train_Category_Y, "color_output": train_Color_Y},
    validation_data=(test_X, {"category_output": test_Category_Y, "color_output": test_Color_Y}),
    epochs=EPOCHS,
    verbose=VERBOSE
)
```

- Outputs structured as dictionaries  
- Recommended callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`  

---

## Evaluation & Visualization

- **Category**:
  - Loss curve: `plot/category_loss.png`
  - Metrics: `plot/category_metrics.png`
  - Confusion matrix: `plot/category_confusion_matrix.png`

- **Color**:
  - Loss curve: `plot/color_loss.png`
  - Metrics: `plot/color_metrics.png`
  - Confusion matrix: `plot/color_confusion_matrix.png`

- **Combined Loss**: `plot/total_loss.png`

Metrics include Accuracy, Precision, Recall, and F1-score.  
Confusion matrices provide class-level insights.

---

## Results

- Multi-task learning improves efficiency and knowledge transfer compared to separate models.  
- Shared features accelerate convergence.  
- Task-specific branches capture unique patterns.  
- Final model saved as: `model/fashion_net.keras`

---

## Repository

Full source code and instructions available at:

[GitHub Repository Link]

---

**Author: Sina Balei**
