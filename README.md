# ml-model
Machine Learning Model for predicting age, gender, and personality of the user

# text
Import numpy, keras, tensrflow, nltk(optional), and set up python jupyter notebooks with traditional conda or pip install, in order to emulate my environment. You can also utilize Google Colab, it may be more convinient. All of the code performing above the baseline is within the working_model folder, all of the other code is a draft that could be improved on or is not good for the provided form of text data, such as doc2vec, which did not perform well on the given textual dataset. To run, follow the format:
- [input directory with test dataset] - [output directory]
Working Models:
Logres over text with preprocessing and LIWC
BERT with simple NN
Ensemble
SVM (low accuracy)

# image
Requires Tensorflow, Keras, Pillow, and Numpy in a Python Environment.
The code provided in GenderImageClassifer is a Convolutional Neural Network that is intended to classify gender.

Training a Model:
Setup a folder for the CNN. Then copy the contents of GenderImageClassifer into this folder. Create a folder that will hold your training set. This folder must have subfolders that are labeled with the classifications.

Ex.
image_postchange
  - male
    - (All Your Male Example Images)
  - female
    - (All Your Female Example Images)

Change the path to the training data (ex. "image_postchange/") in CatergoricalGenderClassiferCNN.py and optionally edit the settings.
Run CatergoricalGenderClassiferCNN.py with Python.

Making Predictions:
Put an image in the same folder as ModelTestCategoricalClassifer.py and make sure that the model that was trained is present there as well (Named "image_model" by default).
Edit the path to the image in ModelTestCategoricalClassifer.py
Run ModelTestCategoricalClassifer.py with Python.
