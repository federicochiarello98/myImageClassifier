import tensorflow as tf

model_dir_path = "models/InceptionV3_320x320/"
saved_model_name = 'Mymodel320'
model_h5 = 'Mymodel320.h5'

model = tf.keras.models.load_model(model_dir_path+model_h5)
model.save(model_dir_path+saved_model_name)

print("Model conversion done!")