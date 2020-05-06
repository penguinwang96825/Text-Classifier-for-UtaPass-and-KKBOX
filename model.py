import tensorflow as tf
import keras.backend as K
from keras import regularizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import InputLayer
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Add
from keras.layers import Concatenate
from keras.layers import ZeroPadding1D
from keras.layers import Dropout
from keras.layers import SpatialDropout1D
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalMaxPool1D
from keras.layers import AveragePooling1D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import SimpleRNN
from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.optimizers import Optimizer
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import RMSprop
from keras.optimizers import Nadam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from gensim.models.word2vec import Word2Vec


# Input parameters
config = {
    # Text parameters
    "MAX_FEATURE": 10000, 
    "MAX_LEN": 64, 
    "EMBED_SIZE": 300, 

    # Convolution parameters
    "filter_length": 3, 
    "nb_filter": 150, 
    "pool_length": 2, 
    "cnn_activation": 'relu', 
    "border_mode": 'same', 

    # RNN parameters
    "lstm_cell": 128, 
    "output_size": 50, 
    "rnn_activation": 'tanh', 
    "recurrent_activation": 'hard_sigmoid', 
    
    # FC and Dropout
    "fc_cell": 128, 
    "dropout_rate": 0.25, 

    # Compile parameters
    "loss": 'binary_crossentropy', 
    "optimizer": 'adam', 

    # Training parameters
    "batch_size": 256, 
    "nb_epoch": 500, 
    "validation_split": 0.50, 
    "shuffle": True
}

def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def train_simple_rnn(x_train, y_train, wv_matrix):
    tf.keras.backend.clear_session()
    
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = SimpleRNN(units=config["output_size"], activation=config["rnn_activation"])(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("="*20, "Start Training Simple RNN", "="*20)

    path = 'weights\{}_weights.hdf5'.format("rnn")
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        x_train, y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model

def train_gru(x_train, y_train, wv_matrix):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = GRU(units=config["output_size"], return_sequences=False)(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("="*20, "Start Training GRU", "="*20)

    path = 'weights\{}_weights.hdf5'.format("gru")
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        x_train, y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model

def train_lstm(x_train, y_train, wv_matrix):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = LSTM(units=config['lstm_cell'], return_sequences=False)(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("="*20, "Start Training LSTM", "="*20)

    path = 'weights\{}_weights.hdf5'.format("lstm")
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        x_train, y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model

def train_bilstm(x_train, y_train, wv_matrix):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = Bidirectional(LSTM(units=config['output_size'], return_sequences=True))(x)
    x = Bidirectional(LSTM(units=config['output_size'], return_sequences=False))(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(config['fc_cell'])(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("="*20, "Start Training BiLSTM", "="*20)

    path = 'weights\weights.hdf5'
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        x_train, y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model

# Yoon Kim's paper: Convolutional Neural Networks for Sentence Classification
# Reference from https://www.aclweb.org/anthology/D14-1181.pdf
def train_cnn_static(x_train, y_train, wv_matrix, trainable=False):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=trainable)(x_input)
    
    x_conv_1 = Conv1D(filters=config['nb_filter'], 
                      kernel_size=3, 
                      padding=config['border_mode'], 
                      kernel_regularizer=regularizers.l2(3))(x)
    x_conv_1 = BatchNormalization()(x_conv_1)
    x_conv_1 = Activation("relu")(x_conv_1)
    x_conv_1 = MaxPooling1D(pool_size=(config["MAX_LEN"]-3+1), strides=1, padding="valid")(x_conv_1)
    
    x_conv_2 = Conv1D(filters=config['nb_filter'], 
                      kernel_size=4, 
                      padding=config['border_mode'], 
                      kernel_regularizer=regularizers.l2(3))(x)
    x_conv_2 = BatchNormalization()(x_conv_2)
    x_conv_2 = Activation("relu")(x_conv_2)
    x_conv_2 = MaxPooling1D(pool_size=(config["MAX_LEN"]-4+1), strides=1, padding="valid")(x_conv_2)
    
    x_conv_3 = Conv1D(filters=config['nb_filter'], 
                      kernel_size=5, 
                      padding=config['border_mode'], 
                      kernel_regularizer=regularizers.l2(3))(x)
    x_conv_3 = BatchNormalization()(x_conv_3)
    x_conv_3 = Activation("relu")(x_conv_3)
    x_conv_3 = MaxPooling1D(pool_size=(config["MAX_LEN"]-5+1), strides=1, padding="valid")(x_conv_3)
    
    main = Concatenate(axis=1)([x_conv_1, x_conv_2, x_conv_3])
    main = Flatten()(main)
    main = Dropout(config['dropout_rate'])(main)
    main = Dense(units=1)(main)
    main = Activation('sigmoid')(main)
    model = Model(inputs=x_input, outputs=main)

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("="*20, "Start Training CNN", "="*20)

    path = 'weights\{}_weights.hdf5'.format("lstm")
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        x_train, y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model

def train_cnn_multichannel(x_train, y_train, wv_matrix, trainable=False):
    tf.keras.backend.clear_session()
    
    # Channel 1
    x_input_1 = Input(shape=(config["MAX_LEN"], ))
    embedding_1 = Embedding(
        wv_matrix.shape[0], 
        wv_matrix.shape[1], 
        weights=[wv_matrix], 
        trainable=trainable)(x_input_1)
    x_conv_1 = Conv1D(
        filters=config['nb_filter'], 
        kernel_size=3, 
        padding=config['border_mode'], 
        kernel_regularizer=regularizers.l2(3))(embedding_1)
    x_conv_1 = BatchNormalization()(x_conv_1)
    x_conv_1 = Activation("relu")(x_conv_1)
    x_conv_1 = MaxPooling1D(pool_size=(config["MAX_LEN"]-3+1), strides=1, padding="valid")(x_conv_1)
    flat_1 = Flatten()(x_conv_1)
    
    # Channel 2
    x_input_2 = Input(shape=(config["MAX_LEN"], ))
    embedding_2 = Embedding(
        wv_matrix.shape[0], 
        wv_matrix.shape[1], 
        weights=[wv_matrix], 
        trainable=trainable)(x_input_2)
    x_conv_2 = Conv1D(
        filters=config['nb_filter'], 
        kernel_size=3, 
        padding=config['border_mode'], 
        kernel_regularizer=regularizers.l2(3))(embedding_2)
    x_conv_2 = BatchNormalization()(x_conv_2)
    x_conv_2 = Activation("relu")(x_conv_2)
    x_conv_2 = MaxPooling1D(pool_size=(config["MAX_LEN"]-3+1), strides=1, padding="valid")(x_conv_2)
    flat_2 = Flatten()(x_conv_2)
    
    # Channel 1
    x_input_3 = Input(shape=(config["MAX_LEN"], ))
    embedding_3 = Embedding(
        wv_matrix.shape[0], 
        wv_matrix.shape[1], 
        weights=[wv_matrix], 
        trainable=trainable)(x_input_3)
    x_conv_3 = Conv1D(
        filters=config['nb_filter'], 
        kernel_size=3, 
        padding=config['border_mode'], 
        kernel_regularizer=regularizers.l2(3))(embedding_3)
    x_conv_3 = BatchNormalization()(x_conv_3)
    x_conv_3 = Activation("relu")(x_conv_3)
    x_conv_3 = MaxPooling1D(pool_size=(config["MAX_LEN"]-3+1), strides=1, padding="valid")(x_conv_3)
    flat_3 = Flatten()(x_conv_3)
    
    main = Concatenate(axis=1)([flat_1, flat_2, flat_3])
    main = Dense(units=100)(main)
    main = Activation('relu')(main)
    main = Dropout(config['dropout_rate'])(main)
    main = Dense(units=1)(main)
    main = Activation('sigmoid')(main)
    model = Model(inputs=[x_input_1, x_input_2, x_input_3], outputs=main)

    model.compile(loss=config["loss"], optimizer=config["optimizer"], metrics=[get_f1])

    print("="*20, "Start Training CNN-multichannel", "="*20)

    path = 'weights\{}_weights.hdf5'.format("lstm")
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        [x_train, x_train, x_train], y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model

def train_cnn_lstm(x_train, y_train, wv_matrix):
    tf.keras.backend.clear_session()
    
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(config['dropout_rate'])(x)
    x = Conv1D(filters=config['nb_filter'], kernel_size=config['filter_length'], padding=config['border_mode'])(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=config['pool_length'])(x)
    x = LSTM(units=config['lstm_cell'], return_sequences=False)(x)
    x = Dense(config['fc_cell']*2)(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(units=1)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=x_input, outputs=x)

    model.compile(loss=config['loss'], optimizer=config['optimizer'], metrics=[get_f1])

    print("="*20, "Start Training CNN-LSTM", "="*20)

    path = 'weights\{}_weights.hdf5'.format("cnn_lstm")
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=0, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    history = model.fit(
        x_train, y_train, 
        batch_size=config['batch_size'], 
        epochs=config['nb_epoch'], 
        validation_split=config['validation_split'], 
        shuffle=config['shuffle'], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model

def identity_resnet_block(x, nb_filter):
    x_shortcut = x

    # First component of main path
    res_x = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)

    # Second component of main path
    res_x = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(res_x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)

    # Third component of main path
    res_x = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(res_x)
    res_x = BatchNormalization()(res_x)

    # Final Step: add shortcut value to the main path
    x = Add()([x_shortcut, res_x])
    output = Activation('relu')(x)
    return output

def convolutional_resnet_block(x, nb_filter):
    x_shortcut = x

    # First component of main path
    res_x = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)

    # Second component of main path
    res_x = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(res_x)
    res_x = BatchNormalization()(res_x)
    res_x = Activation("relu")(res_x)

    # Third component of main path
    res_x = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(res_x)
    res_x = BatchNormalization()(res_x)

    # Shortcut path
    x_shortcut = Conv1D(filters=nb_filter, kernel_size=2, strides=1, padding='same')(x_shortcut)
    x_shortcut = BatchNormalization()(x_shortcut)

    # Final Step: add shortcut value to the main path
    x = Add()([x_shortcut, res_x])
    output = Activation('relu')(x)
    return output

def train_text_resnet(x_train, y_train, wv_matrix):
    tf.keras.backend.clear_session()
    # Resnet for reviews of UtaPass and KKBOX
    x_input = Input(shape=(config["MAX_LEN"], ))
    x = Embedding(wv_matrix.shape[0], wv_matrix.shape[1], weights=[wv_matrix], trainable=False)(x_input)
    x = SpatialDropout1D(0.25)(x)

    # Stage 1
    x = Conv1D(filters=64, kernel_size=3, strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    # Stage 2
    x = convolutional_resnet_block(x, 64)
    x = identity_resnet_block(x, 64)
    x = identity_resnet_block(x, 64)
    # Stage 3
    x = convolutional_resnet_block(x, 128)
    x = identity_resnet_block(x, 128)
    x = identity_resnet_block(x, 128)
    x = identity_resnet_block(x, 128)
    # Stage 4
    x = convolutional_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    x = identity_resnet_block(x, 256)
    # Stage 5
    x = convolutional_resnet_block(x, 512)
    x = identity_resnet_block(x, 512)
    x = identity_resnet_block(x, 512)
    # Average pool
    x = AveragePooling1D(pool_size = 1)(x)
    # Output layer
    x = Flatten()(x)

    x = Dense(units=1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(units=1024, activation='relu')(x)
    x = Dense(units=1)(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=x_input, outputs=x)
    model.compile(loss=config["loss"],
                  optimizer=config["optimizer"],
                  metrics=[get_f1])
    
    path = r'C:\Users\YangWang\Desktop\Text_Classifier_for_UtaPass_and_KKBOX\notebook\weights\text_resnet_weights.hdf5'
    # model_checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor = 'loss', patience=100, verbose=1, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=50, min_lr=0.001)
    
    history = model.fit(
        x_train, y_train, 
        batch_size=config["batch_size"], 
        epochs=config["nb_epoch"], 
        validation_split=config["validation_split"], 
        shuffle=config["shuffle"], 
        verbose=0, 
        callbacks=[early_stopping, reduce_lr])
    
    return history, model