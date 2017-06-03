from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

from models.models import create_model
import util.util as util

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

        self.parser.add_argument('--real_A', type=str, required=True)
        self.parser.add_argument('--fake_B', type=str, required=True)
        self.isTrain = False


def main():
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

    # Load image
    real = Image.open(opt.real_A)
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
    util.save_image(fake, opt.fake_B)


if __name__ == '__main__':
    main()
