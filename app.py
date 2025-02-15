
import numpy as np
import tensorflow as tf
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Load the trained model
model = tf.keras.models.load_model("digit_model.h5")

# Streamlit UI
def main():
    st.title("Digit Recognition")
    st.write("Draw a digit (0-9) below:")

    # Create a drawing canvas
    canvas_result = st_canvas(
        fill_color="black",  # Background color
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28,28))
        img = img / 255.0  # Normalize
        img = img.reshape(1, 28, 28, 1)
        
        prediction = model.predict(img)
        st.write("Predicted Digit:", np.argmax(prediction))

if __name__ == "__main__":
    main()
    