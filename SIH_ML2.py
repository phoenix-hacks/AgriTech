
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model


def build_unet(input_shape):
    inputs = tf.keras.Input(input_shape)
    
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    
    u1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(p1)
    u1 = concatenate([u1, c1])
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(u1)
    
    model = Model(inputs, outputs)
    return model


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not read the image at {image_path}. Check the file path.")
    image = cv2.resize(image, target_size) / 255.0  # Normalize to [0, 1]
    return image


def load_dataset(input_paths, marked_paths, target_size=(128, 128)):
    input_images = []
    marked_images = []
    
    for input_path, marked_path in zip(input_paths, marked_paths):
        # Load and preprocess input image
        input_image = load_and_preprocess_image(input_path, target_size)
        input_images.append(input_image)
        
        # Load and preprocess marked image
        marked_image = load_and_preprocess_image(marked_path, target_size)
        marked_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
        marked_images.append(marked_image)
    
    # Convert lists to numpy arrays
    input_images = np.array(input_images)
    marked_images = np.array(marked_images)
    
    # Expand dimensions to match model input shape
    input_images = np.expand_dims(input_images, axis=-1) if input_images.shape[-1] != 3 else input_images
    marked_images = np.expand_dims(marked_images, axis=-1)  # Shape: (num_samples, 128, 128, 1)
    
    return input_images, marked_images


# Define paths to your input and marked images
input_image_paths = ['./Gujarat.jpeg', ]
marked_image_paths = ['./draw1_raw.png', '11.jpg','12.jpg','13.jpg','14.jpg']

# Load and preprocess dataset
input_images, marked_images = load_dataset(input_image_paths, marked_image_paths)

# Build and compile the model
model = build_unet((128, 128, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the dataset
model.fit(input_images, marked_images, epochs=5, batch_size=2)

# Load a test image for prediction
test_image = load_and_preprocess_image('./gautambudhnagar_img.jpeg')
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

# Predict the marked areas on the test image
predicted_mask = model.predict(test_image)[0]
predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask

# Save the predicted mask as an output image
cv2.imwrite('output_marked_image.jpg', predicted_mask)
print("Processed image saved as 'output_marked_image.jpg'.")


