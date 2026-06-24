import tensorflow as tf
import json

model_path = 'thesis_pipeline/models/lstm.h5'
model = tf.keras.models.load_model(model_path)
print("Model name:", model.name)
for layer in model.layers:
    print(f"Layer: {layer.name}, config: {layer.get_config()}")
