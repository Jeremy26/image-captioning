# Image Captioning 

### Intro to Deep Learning course in the Advanced Machine Learning specialization.

In this project, we combine CNNs and RNNs to generate captions given images.

We use a combination of a CNN encoder to compute convolutional features and layers of LSTM cells as a decoder.

![](https://camo.githubusercontent.com/9a6daff6d4cf95592fc3d1871670d32a87b1bb9f/68747470733a2f2f6769746875622e636f6d2f6873652d616d6c2f696e74726f2d746f2d646c2f626c6f622f6d61737465722f7765656b362f696d616765732f656e636f6465725f6465636f6465722e706e673f7261773d31)


## DATASET
![data](https://miro.medium.com/max/1068/1*u5lzqQYD4LHrBTywNOkJng.png)

The dataset is a collection of **images** and **captions**.
For each image, a set of sentences is used as a label to describe the scene.

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

## DECODER
The decoder part is using Recurrent Neural Networks and LSTM cells to generate the captions.

![](https://miro.medium.com/max/4744/1*ERwScS7k6IH3hZIJmGdHDg.png)

The CNN output is fed directly to a Recurrent Neural Network that learn to generate the words.
In order to get a long-term memory, the RNN type is full of LSTM cells (Long Short Term Memory) that can keep the state of a word. For example, ` a man holding ___ beer ` could be understood as ` a man holding his beer ` so the notion of masculinity is preserved here.

The decoder part first uses word embeddings.
Let's analyze the function.

We first define a Decoder class and two placeholders.
In tensorflow, as Placeholder is used to feed data into a model when training.
We will have **one placeholder for image embedding and one for the sentences.**
```python
class decoder:
    img_embeds = tf.placeholder('float32', [None, IMG_EMBED_SIZE])
    sentences = tf.placeholder('int32', [None, None])
```
Then, we define functions: 
* img_embed_to_bottleneck will reduce the number of parameters.
* img_embed_bottleneck_to_h0 will convert the previously gotten image embedding into the initial LSTM cell
* word_embed will create a word embedding layer: the length of the vocabulary (all existing words)

```python
    img_embed_to_bottleneck = L.Dense(IMG_EMBED_BOTTLENECK, input_shape=(None, IMG_EMBED_SIZE), activation='elu')
    img_embed_bottleneck_to_h0 = L.Dense(LSTM_UNITS,input_shape=(None, IMG_EMBED_BOTTLENECK),activation='elu')
    word_embed = L.Embedding(len(vocab), WORD_EMBED_SIZE)
```

* The next part creates an LSTM cell of a few hundred units.
* Finally, the network must predict words. We we call these predictions logits and we thus need to convert the LSTM output into logits:
* token_logits_bottleneck convert the LSTM to logits bottleneck. That reduces the model complexity
* token_logits convert the bottleneck features into logits using a `Dense()` layer

```python
    lstm = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)
    token_logits_bottleneck = L.Dense(LOGIT_BOTTLENECK, input_shape=(None, LSTM_UNITS), activation="elu")
    token_logits = L.Dense(len(vocab), input_shape=(None, LOGIT_BOTTLENECK))
```
* We can then condition our LSTM cell on the image embeddings placeholder.
* We embed all the tokens but the last
* Then, we create a **dynamic RNN** and calculate token logits for all the hidden states. We will use this with the grounth truth.
* We create a loss mask that will take the value 1 for real tokens and 0 otherwise
* Finally, we compute a **cross-entropy loss**, generally used for classification. This loss is used to compare the `flat_ground_truth` to the `flat_token_logits` (prediction).
```python
    c0 = h0 = img_embed_bottleneck_to_h0(img_embed_to_bottleneck(img_embeds))
    word_embeds = word_embed(sentences[:, :-1])
    hidden_states, _ = tf.nn.dynamic_rnn(lstm, word_embeds,initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0))

    flat_hidden_states = tf.reshape(hidden_states, [-1, LSTM_UNITS])
    flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states))
    flat_ground_truth = tf.reshape(sentences[:, 1:], [-1])

    flat_loss_mask = tf.not_equal(flat_ground_truth, pad_idx)
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_ground_truth, logits=flat_token_logits)
    loss = tf.reduce_mean(tf.boolean_mask(xent, flat_loss_mask))
```
