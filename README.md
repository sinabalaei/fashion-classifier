````markdown
# Multi-Task Fashion Classification Using Keras Functional API

This project implements a **multi-task convolutional neural network (CNN)** to simultaneously classify the **category** and **color** of clothing images. The network is designed using the **Keras Functional API**, allowing flexible architecture with shared inputs and multiple task-specific outputs. Separate losses, configurable loss weights, and structured evaluation metrics ensure robust multi-task learning.

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

The dataset contains clothing images labeled with both **category** and **color**. All images are resized to **96×96 pixels** and normalized to `[0,1]`.

| Category & Color | Number of Images |
|-----------------|-----------------|
| black_jeans     | 344             |
| black_jeans     | 358             |
| blue_dress      | 386             |
| blue_jeans      | 356             |
| blue_shirt      | 369             |
| red_dress       | 380             |
| red_shirt       | 332             |

- **Training/Test Split**: 80% training, 20% testing.  
- Each image consistently retains both category and color labels to preserve alignment across tasks.  

> **Note**: Dataset source is custom; images organized as `color_category/imagename.jpg`.

---

## Project Overview

- **Objective**: Predict clothing **category** (e.g., jeans, dress, shirt) and **color** (e.g., red, blue, black) simultaneously.  
- **Approach**: Multi-task CNN using **Keras Functional API**.  
- **Advantages**:
  - Shared feature extraction reduces computation.  
  - Task-specific branches allow independent learning of category and color.  
  - Separate losses for better optimization and interpretability.  

---

## Keras Functional API

The **Functional API** allows flexible network definitions with:

- Multiple inputs  
- Multiple outputs  
- Shared layers  
- Non-linear computation graphs  

Unlike the Sequential API, layers are treated as functions operating on tensors. In this project, it enables a **shared input layer** and **two separate output branches** for category and color:

```python
net = models.Model(
    inputs=input_layers,
    outputs=[cat_net, col_net],
    name="fashionNet"
)
````

This structure allows shared feature extraction at the input level while preserving **task-specific branches** for each prediction.

---

## Label Encoding

`LabelBinarizer` from `scikit-learn` is used to convert string labels to one-hot vectors:

* **Advantages**:

  * Automatically handles string labels
  * Outputs compatible with `categorical_crossentropy`
  * Simple and reliable for multi-class classification

**Alternatives**:

| Method         | Notes                                                                      |
| -------------- | -------------------------------------------------------------------------- |
| LabelEncoder   | Produces integer labels; requires conversion for categorical cross-entropy |
| OneHotEncoder  | Flexible for pipelines but more complex                                    |
| to_categorical | Requires integer labels; suitable when labels are numeric                  |

`LabelBinarizer` is the practical choice for this project.

---

## Data Splitting

Dataset is split using `train_test_split`:

```python
train_test_split(
    all_images,
    category_labels,
    color_labels,
    test_size=0.2,
    random_state=42
)
```

* Ensures **80% training** and **20% testing**
* Maintains consistent pairing of images with both category and color labels

---

## Multi-Output Network

The network takes a **single image input** and produces two outputs:

* `category_output`: predicts clothing category
* `color_output`: predicts clothing color

Each branch has independent convolutional and dense layers to learn task-specific features while sharing early representations. This approach improves efficiency and encourages knowledge transfer between related tasks.

---

## Loss Functions & Weighting

**Separate losses** are assigned for each output:

```python
losses = {
    "category_output": "categorical_crossentropy",
    "color_output": "categorical_crossentropy"
}
```

* **Benefits**:

  * Independent optimization for each task
  * Clear interpretability of training dynamics
  * Easier debugging and analysis

**Loss weights** allow balancing contributions of each task:

```python
loss_weights = {"category_output": 1.0, "color_output": 1.0}
```

* **Purpose**:

  * Prevent one task from dominating
  * Improve convergence stability
  * Adjust for tasks with different difficulty or loss scales

**Weighted total loss formula**:

```
L_total = L1 + λ * L2
```

* `λ` can be tuned based on validation performance.
* Practical strategies:

  * Manual tuning with grid search
  * Normalization using initial loss magnitudes
  * Dynamic weighting approaches (GradNorm, uncertainty weighting, DWA)

---

## Training

Training is performed with `model.fit`:

```python
history = model.fit(
    x=train_X,
    y={"category_output": train_Category_Y, "color_output": train_Color_Y},
    validation_data=(test_X, {"category_output": test_Category_Y, "color_output": test_Color_Y}),
    epochs=EPOCHS,
    verbose=VERBOS
)
```

* Outputs and validation data are structured as dictionaries matching output layer names.
* Keras automatically assigns corresponding losses and weights.
* Recommended callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.

---

## Evaluation & Visualization

* **Category Predictions**:

  * Loss: `plot/category_loss.png`
  * Metrics: `plot/category_metrics.png`
  * Confusion Matrix: `plot/category_confusion_matrix.png`

* **Color Predictions**:

  * Loss: `plot/color_loss.png`
  * Metrics: `plot/color_metrics.png`
  * Confusion Matrix: `plot/color_confusion_matrix.png`

* **Total Loss** (combined): `plot/total_loss.png`

Metrics are computed using **Accuracy, Precision, Recall, F1-score**, and confusion matrices provide detailed class-wise analysis.

---

## Results

* Multi-task learning improves **efficiency** and **knowledge transfer** compared to independent models.

* Shared features enhance convergence, while task-specific branches capture unique patterns.

* Model and metrics are saved for deployment and further evaluation:

* **Model File**: `model/fashion_net.keras`

---

## Repository

Full source code and instructions are available at:

[Your GitHub Repository Link Here]

---

**Sina Balei**

