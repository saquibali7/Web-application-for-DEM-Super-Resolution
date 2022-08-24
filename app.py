import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, session, send_from_directory
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


transform2 = A.Compose([
       A.Resize(128, 128),
       ToTensorV2(),
])

transform2 = transforms.Compose([
    transforms.PILToTensor()
])


from flask_wtf import FlaskForm

app = Flask(__name__)

img_arr= np.random.random((128,128))

app.config['SECRET_KEY'] = 'jwfhiuwehfuie'
app.config['UPLOADED_PHOTOS_DEST'] = 'Images'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

class MyForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Images only!'),
            FileRequired('File was empty!')
        ]
    )
    submit = SubmitField('Upload')
    

@app.route('/upload/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = MyForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        img_arr = np.array(Image.open(f'Images/{filename}'))
        # img = transform2(img_arr)
        im = Image.fromarray(img_arr)
        im = im.convert("L")
        im.save("Images/gen.jpeg")
        # img = transform(image=img_arr)['image']

        
        # out = model(img)
        file_url = url_for('get_file', filename=filename)
        file_url1 = url_for('get_file', filename='gen.jpeg')
        return render_template('index.html', form=form, file_url = file_url, file_url1=file_url1)
        
    else :
        file_url = None
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)      