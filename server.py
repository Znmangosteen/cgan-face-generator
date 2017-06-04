from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from models.models import create_model
import util.util as util

import os
import time

from flask import Flask, jsonify, request, redirect, url_for, send_file, make_response
from flask_cors import CORS, cross_origin

import base64
import datetime

## Model options
from options.base_options import BaseOptions

class GenOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.isTrain = False

# Parse argument options
opt = GenOptions().parse()
# Some defaults values for generating purpose
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.use_dropout = True
opt.align_data = True
opt.model = 'one_direction_test'
opt.which_model_netG = 'unet_256'
opt.which_direction = 'AtoB'

# Load model
model = create_model(opt)

# Config
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'upload')
app.config['GAN_FOLDER'] = os.path.join(os.getcwd(), 'cgan')
#app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png'])


# Helpers
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def error(msg):
    return jsonify({'error': msg})

# Routers
@app.route('/')
def pong():
    return 'Pong', {'Content-Type': 'text-plain; charset=utf-8'}

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return error('file form-data not existed'), 412

    image = request.files['file']
    #if not allowed_file(image.filename):
        #return error('Only supported %s' % app.config['ALLOWED_EXTENSIONS']), 415

    # Submit taylor.jpg ---> taylor_1234567.jpg (name + timestamp)
    image_name, ext = image.filename.rsplit('.', 1)
    image_name = image_name + '_' + str(int(time.time())) + '.' + ext
    # Save image to /upload
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)

    return send_file(image_path)

@app.route('/gen_base64', methods=['POST'])
def gen_base64():
    if 'image' not in request.json:
        return error('Stupid request'), 412

    image_data = request.json['image']

    ip = request.remote_address
    timestamp = datetime.datetime.now().isoformat()
    image_name = ip + timestamp + '.png'

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

    with open(image_path, "wb") as f:
        f.write(base64.decodebytes(image_data))

    ## Load image and begin generating
    real = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Scale(opt.loadSize),
        transforms.RandomCrop(opt.fineSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # Load input
    input_A = preprocess(real).unsqueeze_(0)
    model.input_A.resize_(input_A.size()).copy_(input_A)
    # Forward (model.real_A) through G and produce output (model.fake_B)
    model.test()

    # Convert image to numpy array
    fake = util.tensor2im(model.fake_B.data)
    output_path = os.path.join(app.config['GAN_FOLDER'], image_name)
    # Save image
    util.save_image(fake, output_path)

    return send_file(output_path)

    

@app.route('/gen', methods=['POST'])
def gen():
    if 'file' not in request.files:
        return error('file form-data not existed'), 412

    image = request.files['file']
  
    # Submit taylor.jpg ---> taylor_1234567.jpg (name + timestamp)
    image_name, ext = image.filename.rsplit('.', 1)
    image_name = image_name + '_' + str(int(time.time())) + '.' + ext
    # Save image to /upload
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image.save(image_path)


    ## Load image and begin generating
    real = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Scale(opt.loadSize),
        transforms.RandomCrop(opt.fineSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # Load input
    input_A = preprocess(real).unsqueeze_(0)
    model.input_A.resize_(input_A.size()).copy_(input_A)
    # Forward (model.real_A) through G and produce output (model.fake_B)
    model.test()

    # Convert image to numpy array
    fake = util.tensor2im(model.fake_B.data)
    output_path = os.path.join(app.config['GAN_FOLDER'], image_name)
    # Save image
    util.save_image(fake, output_path)

    return send_file(output_path)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
