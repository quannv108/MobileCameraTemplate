import os
import sys
import time
sys.path.insert(0,'..')

from pathlib import Path
import numpy as np
import json
from shutil import copyfile, copytree, rmtree
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify
from PIL import Image
import matplotlib.pyplot as plt


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_img'
app._static_folder = os.path.abspath("static")

# TODO: load model here
model = None

@app.route('/static/<path:filename>')
def serve_static(filename):
	'''
	serve static file for this project
	Args:
	- filename: string, the file request get from url
	Another information can get from global variables: request
	Return:
	- static content (js, image, html, css, etc....)
	Raise: None
	'''
	root_dir = app.root_path
	return send_from_directory(os.path.join(root_dir, 'static'), filename)

@app.route('/', methods=['GET'])
def index():
	'''
	this function return response for any request from home page
	including POST and GET
	Args: None
	Return:
	- html: Response object, contain text value inside
	Raise: None
	'''
	return render_template('index.htm')


@app.route('/predict', methods=['POST'])
def predict_uploaded_image():
	'''
	handle upload image POST request,
	1. check upload file existed
	2. save to upload folder
	3. gray scale and make input for model prediction
	4. prediction
	5. return to user the result of predict
	Args: None
	- file uploaded can be check in global variables: request
	Return:
	- response: Response object, contain text value inside
	Raise: None
	'''
	result = {
	"image": "",
	"class": -1,
	"confidence": 0,
	"eslapse": 0
	}
	
	 # check if the post request has the file part
	if 'pic' not in request.files:
		return jsonify({
			'error': -1,
			'msg': 'no image to process'
		})
	file = request.files['pic']
	# if user does not select file, browser also
	# submit an empty part without filename
	if file.filename == '':
		return jsonify({
			'error': -2,
			'msg': 'image have no filename'
		})
	if file:
		# save file to upload folder
		start_time = time.time()
		filename = secure_filename(file.filename)
		filename = filename + str(start_time) + '.jpg'
		fullpath = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
		file.save(fullpath)
		print(type(file))

		print(filename)
		#load file and pre-processing image
		size = 64
		new_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resized_' + filename)
		print('new image will save at: ' + new_path)

		# TODO: pre-process image here
		Image.open(fullpath).resize((size, size)).convert('L').save(new_path)

		# get image and parse to numpy array
		x = plt.imread(new_path).flatten()
		X = np.array([x])

		class_ = 0	# class of object from image
		confidence = 0 # confidence score
		# TODO predict
		# class_, confident = model.predict(X)

		eslape_time = (time.time() - start_time) * 1000

		#add predict result to render dict
		result["class"] = class_
		result["confidence"] = confidence*100
		result["eslapse"] = eslape_time

		#remove old image
		os.remove(new_path)

		# save image as formated
		millis = int(round(start_time * 1000))
		new_filename = "{2}_predictted_{0}_{1}_{3}_".format(class_, confidence*100, millis, eslape_time) + filename
		new_path = os.path.join(app.config["UPLOAD_FOLDER"], new_filename)
		os.rename(fullpath, new_path)

		#add image to render dict
		result["image"] = request.url_root + new_path

	#send response
	return jsonify({
		'error': 0,
		'data': result
	})

if __name__ == "__main__":
	'''
	main function if you run this script from terminal
	We suggest that you should start server using "flask run" command instead this way
	'''
	app.run()
