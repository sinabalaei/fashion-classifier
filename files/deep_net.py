import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

from tensorflow.keras import models, layers
from config import Image_shape

class FashionNet():

    # Static method to build the multi-output CNN model
    @staticmethod
    def build(numberCategory, numberColor):

        # Shared input layer for both outputs
        input_layers = layers.Input(Image_shape)

        # Category branch
        x = layers.Conv2D(32, 3, activation="relu", padding="same")(input_layers)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x) 
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)             
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)  
        x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)             
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

        x = layers.Flatten()(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)    
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(numberCategory)(x)
        cat_net = layers.Activation("softmax", name="category_output")(x)

        # Color branch
        y = layers.Conv2D(16, 3, activation="relu", padding="same")(input_layers)
        y = layers.BatchNormalization()(y)
        y = layers.MaxPooling2D((3, 3))(y)
        y = layers.Dropout(0.25)(y)

        y = layers.Conv2D(32, 3, activation="relu", padding="same")(y)
        y = layers.BatchNormalization()(y)
        y = layers.MaxPooling2D((2, 2))(y)
        y = layers.Dropout(0.25)(y)

        y = layers.Conv2D(32, 3, activation="relu", padding="same")(y)
        y = layers.BatchNormalization()(y)
        y = layers.MaxPooling2D((2, 2))(y)
        y = layers.Dropout(0.25)(y)

        y = layers.Flatten()(y)

        y = layers.Dense(256, activation="relu")(y)
        y = layers.Dropout(0.5)(y)
        y = layers.BatchNormalization()(y)    
        y = layers.Dense(numberColor)(y)
        col_net = layers.Activation("softmax", name="color_output")(y)

        # Define final model with shared input and two outputs
        net = models.Model(inputs = input_layers,
                           outputs = [cat_net, col_net],
                           name = "fashionNet")
        
        return net