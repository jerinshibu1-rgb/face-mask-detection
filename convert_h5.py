import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model("model/cnn_facemask.keras")

print("Saving as H5...")
model.save("facemask.h5")

print("✅ H5 model saved")