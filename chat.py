import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Bidirectional
from keras.models import Model, load_model
from keras.layers import SimpleRNN
from keras.layers import Activation, dot, concatenate
from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse
from flask import Flask, render_template, request, jsonify
import random
import pickle

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:.<>{}`+=~|]", "", text)
    text = " ".join(text.split())
    return text

def transform(encoding, data, vector_size=20):
    """
    :param encoding: encoding dict built by build_word_encoding()
    :param data: list of strings
    :param vector_size: size of each encoded vector
    """
    transformed_data = np.zeros(shape=(len(data), vector_size))
    for i in range(len(data)):
        for j in range(min(len(data[i]), vector_size)):
            try:
                transformed_data[i][j] = encoding[data[i][j]]
            except:
                transformed_data[i][j] = encoding['<un>']
    return transformed_data

def prediction(raw_input,loaded_model):
    INPUT_LENGTH = 15
    OUTPUT_LENGTH = 15
    WORD_CODE_START = 0
    clean_input = clean_text(raw_input)
    input_tok = [nltk.word_tokenize(clean_input)]
    input_tok = [input_tok[0][::-1]]  #reverseing input seq
    encoder_input = transform(encoding, input_tok, 15)
    decoder_input = np.zeros(shape=(len(encoder_input), OUTPUT_LENGTH))
    decoder_input[:,0] = WORD_CODE_START
    for i in range(1, OUTPUT_LENGTH):
        output = loaded_model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
    return output

def decode(decoding, vector):
    """
    :param decoding: decoding dict built by word encoding
    :param vector: an encoded vector
    """
    text = ''
    for i in vector:
        if i == 0:
            break
        text += ' '
        text += decoding[i]
    return text

def get_response(text):
    random_response =['I did not understand','please try again..','i donot have answer for that']
    text = clean_text(text)
    loaded_model = tf.keras.models.load_model('model_attention1.h5')
    output = prediction(text,loaded_model)
    response=decode(decoding, output[0])
    print(response)
    if '<un>' in response and len(response) < 5:
        return random.choice(random_response)
    else:
        return response
    
#https://www.kdnuggets.com/2021/04/deploy-machine-learning-models-to-web.html
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predict',methods=['POST']) 
def predict(): 
    text=request.get_json().get("message")
    response=get_response(text)
    message ={"answer": response}
    return jsonify(message)


if __name__ == '__main__':
    #loading pickle files:
    pickle_off_en = open("Encoding.pickle", 'rb')
    encoding = pickle.load(pickle_off_en)
    pickle_off_de = open("Decoding.pickle", 'rb')
    decoding = pickle.load(pickle_off_de)
    #pickle files loaded........
    app.run()














