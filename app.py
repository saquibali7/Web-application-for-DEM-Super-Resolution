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
import json
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go



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
])


def out_img(img, hr_img):
	hr_img = normalize(hr_img)
	img = normalize(img)
	print(img.shape)
	# k = int(np.random.randint(0,2500))
	# img = img[:,k:k+128,k:k+128]
	# hr_img = hr_img[:,k:k+128,k:k+128]
	
	img  = torch.from_numpy(img)
	hr_img = torch.from_numpy(hr_img)
	img = transformsmain(img)
	hr_img = transformsmain(hr_img)
	print(img.shape)
	print(hr_img.shape)
	return img,  hr_img


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/temp.jpeg'),  code=301)


def normalize(im):
	MIN_H = -500
	MAX_H = 10000
	im = (im - MIN_H)/(MAX_H-MIN_H)
	return im



@app.route('/')
def upload_form():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file1' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file1 = request.files['file1']
	file2 = request.files['file2']
	if file1.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file1 and allowed_file(file1.filename):
		filename1 = secure_filename(file1.filename)
		file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
		img_arr = rasterio.open(f'static/uploads/{filename1}')

		filename2 = secure_filename(file2.filename)
		file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
		img_arr1 = rasterio.open(f'static/uploads/{filename2}')

		hr_img1 = img_arr1.read()
		img = img_arr.read()
		img, hr_img = out_img(img, hr_img1)
		p_img = img
		img = img*255
		img = img.numpy()
		print(img.shape)
		
		img = np.transpose(img, (1,2,0))
		img = img.astype(int)
		cv2.imwrite(f'static/uploads/trial.png', img)
		img = Image.fromarray(img,'RGB')
		img.save("static/uploads/temp.jpeg")
		out_plt = hr_img
		out_im = hr_img
		out_im = np.transpose(out_im, (1,2,0))
		out_im = out_im.numpy()
		out_im = out_im*255
		out_im = out_im.astype(int)
		cv2.imwrite(f'static/uploads/out.png', out_im)
		flash('Image successfully uploaded and displayed below')
		









		x = np.linspace(0,1,512)
		y = np.linspace(0,1,512)


		fig1 = go.Figure(data=[go.Surface(z=out_plt[0], x=x, y=y)])
		fig1.update_layout( autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
		
		fig1.update_layout(
            scene = dict(xaxis_title='X AXIS TITLE',
                yaxis_title='Y AXIS TITLE',
                zaxis_title='Height',
                zaxis = dict(nticks=4, range=[0,1],),),
            width=700,
            margin=dict(r=20, l=10, b=10, t=10))
		x = np.linspace(0,1,512)
		y = np.linspace(0,1,512)


		fig3 = go.Figure(data=[go.Surface(z=p_img[0], x=x, y=y)])
		fig3.update_layout( autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
		
		fig3.update_layout(
            scene = dict(xaxis_title='X AXIS TITLE',
                yaxis_title='Y AXIS TITLE',
                zaxis_title='Height',
                zaxis = dict(nticks=4, range=[0,1],),),
            width=700,
            margin=dict(r=20, l=10, b=10, t=10))		


		x = np.linspace(0,1,512)
		y = np.linspace(0,1,512)
        
		hr_img = hr_img*255
		hr_img = hr_img.numpy()
		print(hr_img.shape, out_plt.shape)
		hr_img = hr_img.astype(int)
		gr_out = out_plt - p_img

		fig2 = go.Figure(data=[go.Surface(z=gr_out[0], x=x, y=y)])
		fig2.update_layout( autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
		
		fig2.update_layout(
            scene = dict(xaxis_title='X AXIS TITLE',
                yaxis_title='Y AXIS TITLE',
                zaxis_title='Height',
                zaxis = dict(nticks=4, range=[0,1],),),
            width=700,
            margin=dict(r=20, l=10, b=10, t=10))	
			
		graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
		graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
		graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
        
		return render_template('index.html', filename='temp.jpeg', graphJSON1=graphJSON1, graphJSON2=graphJSON2, graphJSON3=graphJSON3)
		
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif, tiff')
		return redirect(request.url)
	return	render_template('index.html')

	





if __name__ == "__main__":
    app.run()