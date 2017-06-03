from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from options.test_options import TestOptions

from models.models import create_model
import util.util as util


def main():
    # Parse argument options
    opt = TestOptions().parse()
    # Some defaults values for generating purpose
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.use_dropout = True
    opt.align_data = True
    opt.model = 'pix2pix'
    opt.which_model_netG = 'unet_256'
    opt.which_direction = 'AtoB'

    # Load model
    model = create_model(opt)

    # Load image
    real = Image.open('./real_A.jpg')
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
    # Save image
    util.save_image(fake, './fake_B.png')


if __name__ == '__main__':
    main()
