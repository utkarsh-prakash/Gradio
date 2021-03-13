import tensorflow as tf
import numpy as np
import pandas as pd
import gradio as gr

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
model = tf.keras.models.load_model('models/cnn')

def predict(img):
    print("Input recieved")
    img = img.astype("float32") / 255
    img = np.expand_dims(img, -1)
    
    prediction = model.predict(np.asarray([img]))[0]
    labels = list(range(10))
    print("Prediction:",prediction.argmax())
    return {labels[i]: float(prediction[i]) for i in range(10)}
    
if __name__ == "__main__":
    iface = gr.Interface(predict, 
        gr.inputs.Image(image_mode="L", source="canvas", shape=(28, 28), invert_colors=True), 
        gr.outputs.Label(num_top_classes=4),
        server_port=5001, 
        server_name="127.0.0.1")
    iface.launch(share=True)