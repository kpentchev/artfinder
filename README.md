# artfinder

ML project for recognizing art paintings.

Technologies used:
* python 3
* tensorflow

## Install
* `brew install python3`
* `vi ~/.bash_profile` ; `alias python='python3'`; `source ~/.bash_profile`
* `pip3 install tensorflow`
* `pip3 install Augmentor`

## Steps
1. Generate training set using `distort.py`
* input: a folder containing a few source images of the same painting
* output: a folder containing tousands of distorted images of the painting
2. Generate tensorflow input slices using `preprocess.py`
* input: one folder per painting with tousands of images; `label.txt` file with one label per line per image
* output: slices for either training or testing; number and names of slaces depend on number of threads used for preprocessing
3. Train model using `train_cnn.py`
* input: the training slices
* output: a trained model
4. Evaluate model using `train_cnn.py`
* input: a trained model
* output: accuracy score

## Neuronal Network model

A convolutional neuronal network (CNN) with 7 layers:
* 4 convolutional layers
** each with a filter size of 3, step 1, padding 'SAME'
** number of filters are 64, 64, 128 and 256 respectively.
** each is coupled with a pooling layer of size 2, step 2
* 1 flat layer
* 1 fully connected layer
** 2048 outputs
* 1 output layer
** outputs = number of labels

Images are processed in batches of 15 and down/up-sized to 384x384 pixels. For each pixel 3 channels (RGB) are used.

Cost function is `reduce_mean` of `softmax_cross_entropy_with_logits_v2`. `AdamOptimizer` with a learning rate of `1e-4` was used for minimizing the function.
