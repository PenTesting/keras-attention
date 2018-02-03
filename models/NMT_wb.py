import numpy as np
import os
import keras
from keras.layers import Lambda
#from keras import backend as k
from keras.models import Model
from keras.layers import Dense, Embedding, Activation, Permute
from keras.layers import Input, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from models.custom_recurrents import AttentionDecoder

def reshape(tensor,batch_size,pad_length,seq_length):
    #tensor = tensor[:batch_size, :20, ]
    tensor = keras.backend.reshape(tensor, (batch_size, pad_length, seq_length))
    return tensor
def reshape2(tensor,batch_size,pad_length,seq_length):
    tensor=tensor[:batch_size, :pad_length,]
    tensor = keras.backend.reshape(tensor, (batch_size, pad_length, seq_length))
    return tensor
def reshape1(tensor,batch_size,pad_length,seq_length):
    tensor=keras.backend.mean(tensor,axis=2)
    tensor = keras.backend.reshape(tensor, (batch_size, pad_length, seq_length))
    return tensor

def simpleNMT(pad_length=100,batch_size=32,
              n_chars=105,
              n_labels=6,
              embedding_learnable=False,
              encoder_units=256,
              decoder_units=256,
              trainable=True,
              return_probabilities=False):
    """
    Builds a Neural Machine Translator that has alignment attention
    :param pad_length: the size of the input sequence
    :param n_chars: the number of characters in the vocabulary
    :param n_labels: the number of possible labelings for each character
    :param embedding_learnable: decides if the one hot embedding should be refinable.
    :return: keras.models.Model that can be compiled and fit'ed

    *** REFERENCES ***
    Lee, Jason, Kyunghyun Cho, and Thomas Hofmann. 
    "Neural Machine Translation By Jointly Learning To Align and Translate" 
    """
    input1 = Input(shape=(pad_length,), dtype='float32')
    input2 = Input(shape=(443,4), dtype='float32')
    input_embed = Embedding(n_chars, n_chars,
                            input_length=pad_length,
                            trainable=embedding_learnable,
                            name='OneHot1')(input1)
    input_embed2= Embedding(n_chars, 20,
                            input_length=443,
                            trainable=embedding_learnable,
                            name='OneHot2')(input2)
    input_embed2=Lambda(reshape1,arguments={'batch_size':batch_size,'pad_length':pad_length,'seq_length':443},name='lambda3')(input_embed2)
    rnn_encoded = Bidirectional(LSTM(encoder_units, return_sequences=True),
                                name='bidirectional_1',
                                merge_mode='concat',
                                trainable=trainable)(input_embed)

    y1_hat = AttentionDecoder(decoder_units,
                             name='attention_decoder_1',
                             output_dim=n_labels,
                             return_probabilities=return_probabilities,
                             trainable=trainable)(rnn_encoded)
    y2_hat = AttentionDecoder(decoder_units, name='attention_decoder_2',
                              output_dim=443,
                              return_probabilities=return_probabilities,
                              trainable=trainable)(input_embed2)

    #y2_hat=keras.backend.reshape(y2_hat,(batch_size,pad_length,443))
    y2_hat=Lambda(reshape2,arguments={'batch_size':batch_size,'pad_length':pad_length,'seq_length':443},name='lambda2')(y2_hat)
    y1_hat = Lambda(reshape, arguments={'batch_size': batch_size, 'pad_length': pad_length,'seq_length':1523},name='lambda1')(y1_hat)
    #y1_hat = keras.backend.reshape(y1_hat, (batch_size, pad_length,n_chars))

    y_hat = keras.layers.concatenate([y1_hat,y2_hat],axis=2)
    #model=Model(inputs=input1,outputs=y1_hat)
    model = Model(inputs=[input1,input2], outputs=y_hat)

    # return model
    return model


if __name__ == '__main__':
    model = simpleNMT()
    model.summary()
