import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.layers import TFSMLayer
import numpy as np
import joblib
from PIL import Image
import os


# --- Configuration ---
MODEL_PATH = './training_output/generator_final.keras'   # <-- folder, no trailing slash
SCALER_PATH = './training_output/scaler_data.gz'
OUTPUT_IMAGE_PATH = './generated_floorplan.png'


def load_generator_model():
    """Attempt normal `load_model()`. If it fails (SavedModel dir), use TFSMLayer."""
    print(f"Loading model from: {MODEL_PATH}")

    try:
        # Try Keras model load (works only if .keras / .h5 single file)
        model = load_model(MODEL_PATH)
        print("✅ Loaded using load_model()")
        return model

    except Exception as e:
        print("⚠️ load_model() failed, switching to TFSMLayer...")
        print("Error:", e)

        try:
            # Fallback for SavedModel folder (Keras 3 recommended way)
            model = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
            print("✅ Loaded using TFSMLayer()")
            return model
        except Exception as e2:
            print("\n❌ ERROR: Could not load the generator model in any format.")
            print("Reason:", e2)
            return None


def generate_image():
    # --- 1. Load Model and Scaler ---
    generator = load_generator_model()
    if generator is None:
        return

    print(f"Loading scaler data from: {SCALER_PATH}")
    try:
        scaler_data = joblib.load(SCALER_PATH)
        scaler = scaler_data['scaler']
        columns = scaler_data['columns']
        latent_dim = scaler_data['latent_dim']
        image_size = scaler_data['image_size']
    except Exception as e:
        print(f"❌ Error loading scaler data: {e}")
        return

    print("\n✅ Model + Scaler loaded successfully.\n")

    # --- 2. Ask User for Room Counts ---
    print("Please enter required counts for the floor plan:")
    print("--------------------------------------------------")

    user_counts = []
    for col in columns:
        while True:
            try:
                val = int(input(f"Enter count for '{col}': "))
                user_counts.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a whole number.")

    print("\n✅ User input captured.")
    print("--------------------------------------------------")

    # --- 3. Prepare Generator Input ---
    labels = np.array(user_counts).reshape(1, -1)
    normalized_labels = scaler.transform(labels)
    normalized_labels = tf.cast(normalized_labels, tf.float32)

    noise_vector = tf.random.normal([1, latent_dim])

    # --- 4. Generate Image ---
    print("🧠 Generating image using CGAN...")

    try:
        generated_tensor = generator([noise_vector, normalized_labels])
    except Exception:
        # Some TFSMLayer models expect a dict instead of list
        generated_tensor = generator({"input_1": noise_vector, "input_2": normalized_labels})

    generated_image = (generated_tensor[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    # --- 5. Save Output ---
    img = Image.fromarray(generated_image)
    img.save(OUTPUT_IMAGE_PATH)

    print("\n🎉 SUCCESS — Image generated and saved at:")
    print(os.path.abspath(OUTPUT_IMAGE_PATH))


if __name__ == "__main__":
    generate_image()

