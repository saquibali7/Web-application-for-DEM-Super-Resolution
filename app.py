import numpy as np
import os
from PIL import Image
from flask import Flask,flash, render_template, request, redirect, session, send_from_directory
from flask import redirect, send_file, url_for
from flask_session import Session
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from model import MobileSR
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from werkzeug.utils import secure_filename
import torch

model = MobileSR()



UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/display/<filename>')
def display_image(filename):
	# return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
	return redirect(url_for('static', filename='uploads/temp.jpeg'),  code=301)


	
@app.route('/')
def upload_form():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		img = np.array(Image.open(f'static/uploads/{filename}'))
		img = Image.fromarray(img)
		img = img.convert("RGB")
		img.save("static/uploads/temp.jpeg")
		flash('Image successfully uploaded and displayed below')
		print(filename)
		file_url1 = url_for('display_image', filename='temp.jpeg')
		print(file_url1)
		# return render_template('index.html', file_url1=file_url1)
		return render_template('index.html', filename='temp.jpeg')
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif, tiff')
		return redirect(request.url)
	return	render_template('index.html')


if __name__ == "__main__":
    app.run()