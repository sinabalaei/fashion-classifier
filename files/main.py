import cv2
import numpy as np
import glob
from utility import make_data, data_preprocessing, plot_training_history,\
      save_metrics_table, save_confusion_matrix
from deep_net import FashionNet
from config import SAVE_DIR, MODEL_DIR, EPOCHS, VERBOS
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# Inform user that dataset loading is starting
print("[INFO] Loading dataset...")

# Preprocess dataset and split into training and testing sets
category_LB, color_LB, train_X, test_X, train_Category_Y,\
        train_Color_Y, test_Category_Y, test_Color_Y = data_preprocessing()

# Get category and color class names from data
_, category_names, color_names, _, _, = make_data()

# Inform user that model building is starting
print("[INFO] Building Gender Classification Network...")

# Build the multi-output fashion classification model
model = FashionNet.build(len(category_LB.classes_), 
                        len(color_LB.classes_))

# Inform user that loss functions are being defined
print("[INFO] Making loss...")

# Define separate losses for category and color outputs
losses = {
    "category_output": "categorical_crossentropy", 
    "color_output": "categorical_crossentropy"
}

# Assign loss weights to balance contributions from each task
loss_weights = {"category_output": 1.0, "color_output": 1.0}

# Inform user that model compilation is starting
print("[INFO] Compiling model...")

# Compile the model with Adam optimizer, defined losses, and loss weights
model.compile(optimizer="adam",
            loss = losses,
            loss_weights = loss_weights)

# Inform user that training is starting
print("[INFO] Training model...")

# Train the model with training data, validation data, and specified epochs
history = model.fit(x=train_X,
            y = {"category_output": train_Category_Y, 
                 "color_output": train_Color_Y},
            validation_data=(test_X,
                             {"category_output": test_Category_Y, 
                              "color_output": test_Color_Y}),
            epochs=EPOCHS,
            verbose=VERBOS)
# Print keys of training history for inspection
print(history.history.keys())

# Inform user that model saving is starting
print("[INFO] Saving model...")

# Save trained model to disk
model.save(MODEL_DIR / "fashion_net.keras")

# Inform user that plotting training curves is starting
print("[INFO] Plotting training curves...")

# Plot training history including losses and save plots
plot_training_history(history, SAVE_DIR)

# Generate predictions on test data for both category and color
pred_category, pred_color = model.predict(test_X)

# Convert predictions from one-hot to class indices
cat_pred = np.argmax(pred_category, axis=1)
col_pred = np.argmax(pred_color, axis=1)

# Convert true labels from one-hot to class indices
cat_true = np.argmax(test_Category_Y, axis=1)
col_true = np.argmax(test_Color_Y, axis=1)

# Save evaluation metrics tables for both tasks
save_metrics_table(cat_true, cat_pred, "category", SAVE_DIR)
save_metrics_table(col_true, col_pred, "color", SAVE_DIR)

# Save confusion matrices for both tasks
save_confusion_matrix(
    cat_true,
    cat_pred,
    class_names=category_names,
    name="category",
    save_dir=SAVE_DIR
)

save_confusion_matrix(
    col_true,
    col_pred,
    class_names=color_names,
    name="color",
    save_dir=SAVE_DIR
)

# Inform user that all processing is done
print("[INFO] DONE!!!")