import base64
import os
import time

from flask import Flask, jsonify, request, redirect, url_for, send_file, make_response, render_template
from flask_cors import CORS, cross_origin

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util

# Model loader
opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

model = create_model(opt)

# Config
app = Flask(__name__, template_folder='templates', static_url_path='/static/')
CORS(app)
app.config['UPLOAD_DIR'] = os.path.join(os.getcwd(), 'upload')
app.config['RESULT_DIR'] = os.path.join(os.getcwd(), 'results')
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Setup
if not os.path.exists(app.config['UPLOAD_DIR']):
    os.mkdir(app.config['UPLOAD_DIR'])

if not os.path.exists(app.config['RESULT_DIR']):
    os.mkdir(app.config['RESULT_DIR'])


# Helpers
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def error(msg):
    return jsonify({'error': msg})


# Routers
@app.route('/')
def pong():
    return 'Hello', {'Content-Type': 'text-plain; charset=utf-8'}


@app.route('/home')
def home_page():
    return render_template('main_page.html')


@app.route('/gen', methods=['POST'])
def gen():

    # if 'file' not in request.files:
    # if 'file' not in request.files:
    #     return error('file form-data not existed'), 412

    # image = request.files['file']
    # if not allowed_file(image.filename):
    #     return error('Only supported %s' % app.config['ALLOWED_EXTENSIONS']), 415

    # Submit taylor.jpg ---> save image to upload/12345678/taylor.jpg (upload/timestamp/imagename.ext)
    t = int(time.time())
    image_dir = os.path.join(app.config['UPLOAD_DIR'], str(t))
    image_path = os.path.join(image_dir, '1.png')

    os.mkdir(image_dir)

    image_data = base64.b64decode(request.data)
    with open(image_path, 'wb') as f:
        f.write(image_data)

    # image.save(image_path)

    # Prepare data loader
    opt.dataroot = image_dir
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        # Forward
        model.test()

        # Convert image to numpy array
        fake = util.tensor2im(model.fake_B.data)
        # Save image
        result_dir = os.path.join(app.config['RESULT_DIR'], str(t))
        result_path = os.path.join(result_dir, '1.png')
        os.mkdir(result_dir)
        util.save_image(fake, result_path)

    # with open(result_path, 'rb') as img_f:
    #     img_stream = img_f.read()
    #     img_stream = base64.b64encode(img_stream)
    # return img_stream
    return send_file(result_path)


if __name__ == '__main__':
    app.run()
