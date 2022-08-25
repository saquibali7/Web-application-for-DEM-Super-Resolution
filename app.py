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

def out_img(img):
	img = normalize(img)
	print(img.shape)
	# k = int(np.random.randint(0,2500))
	# img = img[:,k:k+128,k:k+128]
	
	img  = torch.from_numpy(img)
	img = transformsmain(img)
	print(img.shape)
	m = nn.UpsamplingBilinear2d(scale_factor=2)

	img = img.unsqueeze(0)
	print(img.shape)
	img = img.float()
	bi_out = m(img)
	out = model(img)
	bi_out=bi_out.numpy()
	out = out.detach().numpy()
	return img, out, bi_out


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
		img, out_im, bi_out = out_img(img)
		img = img[0]
		img = img*255
		img = img.numpy()
		img = np.transpose(img, (1,2,0))
		img = img.astype(int)
		cv2.imwrite(f'static/uploads/trial.png', img)
		img = Image.fromarray(img,'RGB')
		img.save("static/uploads/temp.jpeg")
		out_plt = out_im
		out_im = np.transpose(out_im[0], (1,2,0))
		out_im = out_im*255
		out_im = out_im.astype(int)
		cv2.imwrite(f'static/uploads/out.png', out_im)
		bi_out = np.transpose(bi_out[0], (1,2,0))
		bi_out = bi_out*255
		bi_out= bi_out.astype(int)
		cv2.imwrite(f'static/uploads/bi_out.png', bi_out)
		# out_im = Image.fromarray(out_im, 'RGB')
		# out_im.save("static/uploads/out.jpeg")
		flash('Image successfully uploaded and displayed below')
		

		x = np.linspace(0,1,512)
		y = np.linspace(0,1,512)


		fig = go.Figure(data=[go.Surface(z=out_plt[0][0], x=x, y=y)])
		fig.update_layout( autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
		
		fig.update_layout(
			title='Ground Truth Depth Elevation Map',
            scene = dict(xaxis_title='X AXIS TITLE',
                yaxis_title='Y AXIS TITLE',
                zaxis_title='Height',
                zaxis = dict(nticks=4, range=[0,1],),),
            width=700,
            margin=dict(r=20, l=10, b=10, t=10))
			
		graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)






		return render_template('index.html', filename='temp.jpeg', graphJSON=graphJSON)
		
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif, tiff')
		return redirect(request.url)
	return	render_template('index.html')


if __name__ == "__main__":
    app.run()