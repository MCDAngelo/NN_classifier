import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from keras import backend as K
import sys

# From https://github.com/fchollet/keras/issues/5400
def f1_score2(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))).toDou
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))

import keras.metrics
keras.metrics.f1_score = f1_score
keras.metrics.f1_score_ML = f1_score

seed = 7
np.random.seed(seed)

df = pd.read_csv("fake_vecs.csv", sep=",")
df['cat2'] = df['category'] + '1'

print("------First few values of df-------")
print(df.head(5))

## Separate features and labels and convert to numpy NDFrame
X = df.loc[:,'V1':'V100'].values
Y = df.loc[:,'category'].values

## Create encoder to transform string labels to one-hot encoded vectors
encoder = LabelEncoder()
encoder.fit(Y)
print(Y)
encoded_Y = encoder.transform(Y)
print(encoded_Y)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y)

## Split data into training and test sets (test will actually be validation set)
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size = 0.20, random_state=42)

print("------Shape of training data-------")
print(X_train.shape, Y_train.shape)
print("------Shape of test data-------")
print(X_test.shape, Y_test.shape)

n_categories = np.unique(Y).size

## Log for tensorboard
logTensorboard = TensorBoard(log_dir='./tmp/', histogram_freq=0, write_graph=True, write_images=False)

## Function to create and fit model
def create_model(x_train, y_train, h1_size, h2_size, n_cats):
    ## Create model
    model = Sequential()
    model.add(Dense(h1_size, input_dim=x_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(h2_size, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_cats, kernel_initializer='normal', activation='softmax'))
    ## Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])
    model.fit(x_train, y_train, epochs=50, verbose=2, batch_size=10, validation_split=0.33, callbacks=[logTensorboard])
    return model

model = create_model(X_train, Y_train, h1_size=128, h2_size=128, n_cats=n_categories)


def evaluate_model(nn_model, x_test, y_test):
    print("-----------RESULTS ON FINAL VALIDATION SET-------------")
    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    y_test_cat = np.argmax(np_utils.to_categorical(y_test), axis=1)[:,1]
    y_pred = model.predict_classes(x_test)

    # print("Micro F Score: ", f1_score(y_test_cat, y_pred, average='micro'))
    # print("Macro F Score: ", f1_score(y_test_cat, y_pred, average='macro'))
    # print("Weighted F Score: ", f1_score(y_test_cat, y_pred, average='weighted'))
    print("Precision, Recall, FScore, Support (Macro): ", precision_recall_fscore_support(y_test_cat, y_pred, average='macro'))
    print("Precision, Recall, FScore, Support (Micro): ", precision_recall_fscore_support(y_test_cat, y_pred, average='micro'))
    print("Precision, Recall, FScore, Support (by cat): ", precision_recall_fscore_support(y_test_cat, y_pred, average=None))

    print("Confusion Matrix: ", confusion_matrix(y_test_cat, y_pred))

evaluate_model(model, X_test, Y_test)