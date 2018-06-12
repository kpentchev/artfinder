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
** input: a folder containing a few source images of the same painting
** output: a folder containing tousands of distorted images of the painting
2. Generate tensorflow input slices using `preprocess.py`
** input: one folder per painting with tousands of images; `label.txt` file with one label per line per image
** output: slices for either training or testing; number and names of slaces depend on number of threads used for preprocessing
3. Train model using `train_cnn.py`
** input: the training slices
** output: a trained model
4. Evaluate model using `train_cnn.py`
** input: a trained model
** output: accuracy score
