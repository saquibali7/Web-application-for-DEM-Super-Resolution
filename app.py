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
import rasterio




model = MobileSR()
model_path = 'MobileSR2x_epoch=500.pth/MobileSR2x_epoch=500.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))



UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tif'])

transformsmain = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((128,128)),
])

def out_img(img):
	img = img/10500
	img  = transformsmain(img)
	print(img.shape)
	img = img.unsqueeze(0)
	print(img.shape)
	img = img.float()
	out = model(img)
	out = out.detach().numpy()
	return out


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/display/<filename>')
def display_image(filename):
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
		img_arr = np.array(Image.open(f'static/uploads/{filename}'))
		print(img_arr.shape)
		out_im = out_img(img_arr)
		img = Image.fromarray(img_arr,'RGB')
		img.save("static/uploads/temp.jpeg")
		out_im = np.transpose(out_im[0], (1,2,0))
		out_im = Image.fromarray(out_im, 'RGB')
		out_im.save("static/uploads/out.jpeg")
		flash('Image successfully uploaded and displayed below')
		print(filename)
		file_url1 = url_for('display_image', filename='temp.jpeg')
		print(file_url1)
		return render_template('index.html', filename='temp.jpeg')
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif, tiff')
		return redirect(request.url)
	return	render_template('index.html')


if __name__ == "__main__":
    app.run()