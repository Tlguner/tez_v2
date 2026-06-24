import tensorflow as tf
import numpy as np
import os

model_path = 'thesis_pipeline/models/lstm.h5'
test_features_path = 'thesis_pipeline/DL/runs/run_20260613_031744/X_test.npy'

print("TensorFlow version:", tf.__version__)
print("Checking model file existence:", os.path.exists(model_path))
print("Checking test features existence:", os.path.exists(test_features_path))

try:
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    print(model.summary())
    
    print("Loading test features...")
    X_test = np.load(test_features_path)
    print("Test features shape:", X_test.shape)
    
    print("Running prediction...")
    pred = model.predict(X_test[:5])
    print("Predictions:", pred)
except Exception as e:
    import traceback
    traceback.print_exc()
