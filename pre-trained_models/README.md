## When loading a pre-trained model, both "_sp1.h5" and "sp2_.h5" models should loaded.
- xxxx_sp1.h5 : the trained model to predict keypoints whose indices are [0, 1, 2, 3, 20, 21, 28, 29]
- xxxx_sp2.h5 : the trained model for the rest of facial keypoints

## Use the following code
model_sp1 = tf.keras.models.load_model(model_sp1_h5_path, custom_objects={'KerasLayer':hub.KerasLayer, 'rmse':rmse})
model_sp2 = tf.keras.models.load_model(model_sp2_h5_path, custom_objects={'KerasLayer':hub.KerasLayer, 'rmse':rmse})
