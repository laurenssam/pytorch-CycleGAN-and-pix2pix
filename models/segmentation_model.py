import torch
from PIL import Image

from util.transforms import denormalize
from util.util import mask_to_img, tensor2im
from .base_model import BaseModel
from . import networks


class SegmentationModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            pass
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_CE']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['rgb', 'label_img', 'prediction_img']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      opt.init_type, opt.init_gain, self.gpu_ids)
        self.num_classes = opt.output_nc
        if self.isTrain:
            # define loss functions
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255, weight=opt.class_weights.to(self.device))
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, weight_decay=1e-5)
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.input_image = input['A'].to(self.device)
        self.label = input['B'].to(self.device)
        self.image_paths = None

    def compute_visuals(self):
        self.rgb = denormalize(self.input_image[0]).unsqueeze(dim=0)
        self.prediction_img = mask_to_img(torch.argmax(self.prediction.cpu(), dim=1)[0])
        self.label_img = mask_to_img(self.label.cpu()[0])

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.prediction = self.netG(self.input_image)  # G(A)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_G_CE = self.criterion(self.prediction, self.label)
        # combine loss and calculate gradients
        self.loss_G_CE.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        ## Todo: Change network architecture



