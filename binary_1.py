import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

def load_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image at {image_path}")
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image.astype(np.float32)

def load_mask(mask_path, target_size=(128, 128)):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask at {mask_path}")
    mask = cv2.resize(mask, target_size)
    mask = mask / 255.0  # Normalize to [0, 1]
    return mask.astype(np.float32).reshape(target_size[0], target_size[1], 1)

def load_dataset(image_dir, mask_dir, target_size=(128, 128)):
    images = []
    masks = []
    image_filenames = (os.listdir(image_dir))
    mask_filenames = (os.listdir(mask_dir))
    
    for img_file, mask_file in zip(image_filenames, mask_filenames):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        image = load_image(img_path, target_size)
        mask = load_mask(mask_path, target_size)
        
        images.append(image)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Load the dataset
image_dir = 'C:\\Users\\GIRIDHAR\\OneDrive\\Desktop\\data\\images'
mask_dir = 'C:\\Users\\GIRIDHAR\\OneDrive\\Desktop\\data\\masks'
X, y = load_dataset(image_dir, mask_dir)

# Print shapes for verification
print(f"Images shape: {X.shape}")
print(f"Masks shape: {y.shape}")

def build_unet_model(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    
    # Decoder
    u1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = concatenate([u1, c2])

    u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u1)
    u2 = concatenate([u2, c1])
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u2)
    
    model = Model(inputs, outputs)
    return model

# Build the model
model = build_unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Split the dataset into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=4
)
# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()
# Load a new test image
test_image_path = 'C:\\Users\\GIRIDHAR\\OneDrive\\Desktop\\data\\gautambudhnagar_img.jpeg'
test_image = load_image(test_image_path)
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

# Predict the mask
predicted_mask = model.predict(test_image)[0]
predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

# Visualize the original image and predicted mask
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.imread(test_image_path)[:, :, ::-1])
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask.squeeze(), cmap='gray')
plt.title('Predicted Mask')
plt.show()
# Convert the predicted mask to 3 channels for saving as an image
predicted_mask_rgb = np.stack((predicted_mask.squeeze(),)*3, axis=-1) * 255
cv2.imwrite('output_predicted_mask.jpg', predicted_mask_rgb)
print("Predicted mask saved as 'output_predicted_mask.jpg'.")
