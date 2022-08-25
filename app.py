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
import torch.nn as nn
import rasterio
import cv2



model = MobileSR()
model_path = 'MobileSR2x_epoch=500.pth/MobileSR2x_epoch=500.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))



UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 4096 * 4096


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tif'])

transformsmain = transforms.Compose([
	transforms.Resize((128,128)),
	transforms.ToTensor(),
])

def out_img(img):
	img = normalize(img)
	print(img.shape)
	k = int(np.random.randint(0,2500))
	img = img[:,k:k+128,k:k+128]
	img  = torch.from_numpy(img)
	print(img.shape)
	img = img.unsqueeze(0)
	print(img.shape)
	img = img.float()
	out = model(img)
	out = out.detach().numpy()
	return img, out


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/temp.jpeg'),  code=301)


def normalize(im):
	MIN_H = im.min()
	MAX_H = im.max()
	im = (im - MIN_H)/(MAX_H-MIN_H)
	return im


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
		img_arr = rasterio.open(f'static/uploads/{filename}')
		img = img_arr.read()
		img, out_im = out_img(img)
		img = img[0]
		img = img*255
		img = img.numpy()
		img = np.transpose(img, (1,2,0))
		img = img.astype(int)
		cv2.imwrite(f'static/uploads/trial.png', img)
		img = Image.fromarray(img,'RGB')
		img.save("static/uploads/temp.jpeg")
		out_im = np.transpose(out_im[0], (1,2,0))
		out_im = out_im*255
		out_im = out_im.astype(int)
		cv2.imwrite(f'static/uploads/out.png', out_im)
		# out_im = Image.fromarray(out_im, 'RGB')
		# out_im.save("static/uploads/out.jpeg")
		flash('Image successfully uploaded and displayed below')
		return render_template('index.html', filename='temp.jpeg')
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif, tiff')
		return redirect(request.url)
	return	render_template('index.html')


if __name__ == "__main__":
    app.run()