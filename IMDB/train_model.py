from __future__ import print_function
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tempfile
from keras.models import Sequential
from keras.layers import *
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM, Bidirectional
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.utils import shuffle
import numpy as np
import pickle
def bd_lstm(embedding_matrix):
    max_len = 250
    num_classes = 2
    loss = 'binary_crossentropy'
    activation = 'sigmoid'
    embedding_dims = 300
    num_words = 50000
    print('Build word_bdlstm model...')
    model = Sequential()
    model.add(Embedding(  # Layer 0, Start
        input_dim=num_words + 1,  # Size to dictionary, has to be input + 1
        output_dim=embedding_dims,  # Dimensions to generate
        weights=[embedding_matrix],  # Initialize word weights
        input_length=max_len,
        name="embedding_layer",
        trainable=False))
    '''
    model.add(Bidirectional(LSTM(128)))  # 64 / LSTM-2:128 / LSTM-3: 32
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=activation))
    '''
    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(2, activation=activation))
    model.summary()

    model.compile('adam', loss, metrics=['accuracy'])
    return model
def train_text_classifier(x_train,y_train,x_test,y_test,embedding_matrix):

    x_train, y_train = shuffle(x_train, y_train, random_state=0)


    model = bd_lstm(embedding_matrix)
    batch_size = 64
    epochs = 20
    PATIENCE=4
    _, tmpfn = tempfile.mkstemp()
    callbacks = [EarlyStopping(patience=PATIENCE), ModelCheckpoint(
        tmpfn, save_best_only=True, save_weights_only=True)]
    print('Train...')
    print('batch_size: ', batch_size, "; epochs: ", epochs)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True,
              callbacks=callbacks)
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
    print('Saving model weights...')
    model_path='bdlstm_models'
    model.save_weights(model_path)
if __name__ == '__main__':
    f=open('aux_files/dataset_50000.pkl','rb')
    dataset=pickle.load(f)
    embedding_matrix = np.load(('aux_files/embeddings_glove_%d.npy' % (50000)))
    embedding_matrix=embedding_matrix.T
    train_x = pad_sequences(dataset.train_seqs2, maxlen=250, padding='post')
    train_y = np.array(dataset.train_y)



    test_x = pad_sequences(dataset.test_seqs2, maxlen=250, padding='post')
    test_y = np.array(dataset.test_y)
    train_y=np.array([[0,1] if t==1 else [1,0]  for t in train_y])
    #valid_y = np.array([[0, 1] if t == 1 else [1, 0] for t in valid_y])
    test_y = np.array([[0, 1] if t == 1 else [1, 0] for t in test_y])
    print('X_train:', train_x.shape)
    print('y_train:', train_y.shape)
    print('X_test:', test_x.shape)
    print('y_test:', test_y.shape)
    train_text_classifier(train_x,train_y,test_x,test_y,embedding_matrix)
    model = bd_lstm(embedding_matrix)
    model_path = 'bdlstm_models'
    model.load_weights(model_path)
    all_scores_origin = model.evaluate(test_x, test_y)
    print('all origin test_loss: %f, accuracy: %f' % (all_scores_origin[0], all_scores_origin[1]))
