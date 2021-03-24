
# coding: utf-8

# In[ ]:


from flask import Flask, render_template,url_for,request

#ML PACKAGES
import tensorflow.keras
import numpy as np
import pandas as pd
import time
import string
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Embedding, Flatten, Conv1D, SpatialDropout1D, MaxPooling1D,AveragePooling1D, merge, concatenate, Input, Dropout
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.optimizers import SGD
from keras.optimizers import Nadam
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adamax
from tensorflow import keras
import tensorflow as tf

# In[ ]:


app = Flask(__name__)

@app.route('/')
def index():
	df = pd.read_json('data_2000.json')
	dff = df['data'][2]
	#merubah list data menjadi dataframe
	dt=pd.DataFrame(dff,columns=['id_case','class','sentence1'])
	pd.set_option('display.max_rows', dt.shape[0]+1)

	class1 = dt['class'].tolist()
	text = dt['sentence1'].tolist()
	y = dt['class']
	y = to_categorical(y)
	jumdata1 = len(y)
	weak1 = 0
	neut1 = 0
	point1 = 0
	co1 =0

	for i in range(0, jumdata1): 
		if class1[i] == '0' :
		   weak1 = weak1 +1
		elif class1[i] == '2' :
		   neut1 = neut1 +1
		elif class1[i] == '3' :
		   point1 = point1 +1
		elif class1[i] == '1' :
		   co1 = co1 +1

	token = Tokenizer()
	token.fit_on_texts(text)
	sequences = token.texts_to_sequences(text)
	vocab1 = len(token.index_word)+1

	return render_template('index.html', jumdata = jumdata1, vocab = vocab1, weak=weak1, point=point1, co = co1, neut = neut1)


@app.route('/klasifikasi')
def klasifikasi():
	return render_template('klasifikasi.html')

@app.route('/datalatih')
def datalatih():
	df = pd.read_json('data_2000.json')
	dff = df['data'][2]
	dt=pd.DataFrame(dff,columns=['id_case','class','sentence1'])
	sentence1 = [] 
	id_case1 = [] 
	kelas1 = [] 
	# sentence1 = df['sentence1']
	id_case1 = dt['id_case'].tolist()
	sentence1 = dt['sentence1'].tolist()
	kelas1 = dt['class'].tolist()
	# kelas = df['class']
	return render_template('datalatih.html', len = len(id_case1), id_case = id_case1, sentence = sentence1, kelas = kelas1) 

@app.route('/bantuan')
def bantuan():
	return render_template('bantuan.html')

# @app.route('/profile')
# def klasifikasi():
# 	return render_template('klasifikasi.html')

@app.route('/predict', methods=['POST'])
def predict():
	#membaca data json
	df = pd.read_json('view_hasil_1000.json')
	#mengambil kolom data
	dff = df['data'][2]
	#merubah list data menjadi dataframe
	dt=pd.DataFrame(dff,columns=['id_case','class','sentence1'])
	text = [] 
	text = dt['sentence1'].tolist()

	y = dt['class']
	y = to_categorical(y)

	token = Tokenizer()
	token.fit_on_texts(text)
	sequences = token.texts_to_sequences(text)
	
	sequences = np.array(sequences)
	vocab = len(token.index_word)+1

	encoded_text = token.texts_to_sequences(text)

	max_kata = 55
	X = pad_sequences(encoded_text, maxlen = max_kata, padding='post')

	X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state = 40, test_size = 0.3, stratify = y)
	X_train = np.asarray(X_train)
	X_test = np.asarray(X_test)
	Y_train = np.asarray(Y_train)
	Y_test = np.asarray(Y_test)

	# load json and create model
	json_file = open('model1DCNN.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model1DCNN.h5")
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	def get_encoded(x):
		x = token.texts_to_sequences(x)
		x = pad_sequences(x, maxlen = max_kata, padding='post')
		return x
	
	if request.method == 'POST':
		kalimat = []
		kalimat1 = request.form['kalimat']
		kalimat.append(kalimat1)
		x = get_encoded(kalimat)
		y_prob = loaded_model.predict(x) 
	
		my_prediction = y_prob.argmax(axis=-1)

	return render_template('predict.html',prediction = my_prediction, sentence = kalimat1)
if __name__ == '__main__':
	app.run(debug=True)

