# Image Captioning 

### Intro to Deep Learning course in the Advanced Machine Learning specialization.

In this project, we combine CNNs and RNNs to generate captions given images.

We use a combination of a CNN encoder to compute convolutional features and layers of LSTM cells as a decoder.


## ENCODER
The encoder is a Convolutional Neural Network named Inception v3.
This is a popular architecture for image classification.
![alt](https://camo.githubusercontent.com/07c3ce81fec6ae627177a56ad65d23297f58cdf7/68747470733a2f2f6769746875622e636f6d2f6873652d616d6c2f696e74726f2d746f2d646c2f626c6f622f6d61737465722f7765656b362f696d616765732f696e63657074696f6e76332e706e673f7261773d31)

The code used to compute that CNN with Keras is below:
```python
def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input

    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model
```

As you can see, the fully-connected layer is cropped with the parameter `include_top=False` inside the function call.
It means that we directly use the convolutional features and we don't activate them to a purpose (classification, regression, ...).

