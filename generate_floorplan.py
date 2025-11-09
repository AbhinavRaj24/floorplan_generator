import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from PIL import Image
import os
# --- Configuration ---
MODEL_PATH = './training_output/generator_final.keras/'
SCALER_PATH = './training_output/scaler_data.gz'
OUTPUT_IMAGE_PATH = './generated_floorplan.png'

def generate_image():
    # --- 1. Load Model and Scaler ---
    print(f"Loading model from {MODEL_PATH}...")
    try:
        generator = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you run the 'cgan_floorplan_trainer.py' script to completion?")
        return

    print(f"Loading scaler data from {SCALER_PATH}...")
    try:
        scaler_data = joblib.load(SCALER_PATH)
        scaler = scaler_data['scaler']
        columns = scaler_data['columns']
        latent_dim = scaler_data['latent_dim']
        image_size = scaler_data['image_size']
    except Exception as e:
        print(f"Error loading scaler data: {e}")
        print("Did you run the 'cgan_floorplan_trainer.py' script to completion?")
        return

    print("Model and scaler loaded successfully.\n")

    # --- 2. Get User Input ---
    print("Please enter the desired counts for your floor plan:")
    print("--------------------------------------------------")
    
    user_counts = []
    for col in columns:
        while True:
            try:
                count = int(input(f"Enter count for '{col}': "))
                user_counts.append(count)
                break
            except ValueError:
                print("Invalid input. Please enter a whole number.")
    
    print("--------------------------------------------------")

    # --- 3. Prepare Inputs for Generator ---
    
    # Create the label vector
    label_vector = np.array(user_counts).reshape(1, -1)
    
    # Normalize the label vector *exactly* as during training
    normalized_label_vector = scaler.transform(label_vector)
    normalized_label_vector = tf.cast(normalized_label_vector, tf.float32)

    # Create a random noise vector
    noise_vector = tf.random.normal([1, latent_dim])

    # --- 4. Generate the Image ---
    print("Generating floor plan...")
    
    # Pass [noise, label] to the generator
    generated_image_tensor = generator([noise_vector, normalized_label_vector], training=False)
    
    # Denormalize the image from [-1, 1] to [0, 255]
    generated_image = (generated_image_tensor[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    # --- 5. Save the Image ---
    img_pil = Image.fromarray(generated_image)
    img_pil.save(OUTPUT_IMAGE_PATH)
    
    print(f"\nSuccess! Your floor plan has been saved to:")
    print(f"{os.path.abspath(OUTPUT_IMAGE_PATH)}")

if __name__ == "__main__":
    generate_image()
