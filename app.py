from flask import Flask, render_template, request 
from keras import models
from scipy.misc import imsave, imread, imshow
import numpy as np 
import re
import sys
import os
import base64

# Specifing our model loction
sys.path.append(os.path.abspath("./model"))

from load import *

global model, graph
#initialize these variables
model, graph = init()
# print(model,graph)

#decoding an image from base64 into raw representation
def convertImage(imgData1): 
	imgstr = re.search(b'base64,(.*)',imgData1).group(1) 
	with open('output.png','wb') as output: 
		output.write(base64.b64decode(imgstr))

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
	# Making request for the data
	imgData = request.get_data()
	convertImage(imgData)
	print("debug")

	x = imread('output.png',mode='L')
	#compute a bit-wise inversion so black becomes white and vice versa
	x = np.invert(x)
	#make it the right size
	x = imresize(x,(28,28))
	#imshow(x)
	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,28,28,1)
	print("debug2")

	with graph.as_default():
		#perform the prediction
		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		print("debug3")
		#convert the response to a string
		response = np.array_str(np.argmax(out,axis=1))
		return response	

if __name__ == '__main__':
    app.run(debug=False)

