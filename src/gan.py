import numpy as np
import torch
from models import GeneratorUNet, Discriminator
from data_loader_camus import DatasetCAMUS
from torchvision.utils import save_image
import datetime
import time
import sys
RESULT_DIR = 'results'
VAL_DIR = 'val_images'
TEST_DIR = 'test_images'
MODELS_DIR = 'saved_models'


class GAN:
    def __init__(self, data_loader, config, use_wandb, device):

        # Configure data loader
        self.config = config
        self.result_name = config['NAME']
        self.data_loader = data_loader
        self.use_wandb = use_wandb
        self.device = device
        self.step = 0

        # Input shape
        self.channels = config['CHANNELS']
        self.img_rows = config['IMAGE_RES'][0]
        self.img_cols = config['IMAGE_RES'][1]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        assert self.img_rows == self.img_cols, 'The current code only works with same values for img_rows and img_cols'

        # scaling
        self.target_trans = config['TARGET_TRANS']
        self.input_trans = config['INPUT_TRANS']

        # Input images and their conditioning images
        self.conditional_d = config.get('CONDITIONAL_DISCRIMINATOR', False)
        self.recon_loss = config.get('RECON_LOSS', 'basic')

        # Calculate output shape of D (PatchGAN)
        patch_size = config['PATCH_SIZE']
        patch_per_dim = int(self.img_rows / patch_size)
        self.num_patches = (patch_per_dim, patch_per_dim, 1)
        num_layers_D = int(np.log2(patch_size))

        # Number of filters in the first layer of G and D
        self.gf = config['FIRST_LAYERS_FILTERS']
        self.df = config['FIRST_LAYERS_FILTERS']
        self.skipconnections_generator = config['SKIP_CONNECTIONS_GENERATOR']
        self.output_activation = config['GEN_OUTPUT_ACT']
        self.decay_factor_G = config['LR_EXP_DECAY_FACTOR_G']
        self.decay_factor_D = config['LR_EXP_DECAY_FACTOR_D']

        self.generator = GeneratorUNet().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        #self.generator.apply(self.weights_init_normal)
        #self.discriminator.apply(self.weights_init_normal)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=config['LEARNING_RATE_G'],
                                            betas=(config['ADAM_B1'], 0.999))  # 0.0002
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=config['LEARNING_RATE_D'], betas=(config['ADAM_B1'], 0.999))

        self.criterion_GAN = torch.nn.MSELoss().to(self.device)
        self.criterion_pixelwise = torch.nn.L1Loss().to(self.device)  # MAE
        # criterion_pixelwise = torch.nn.L1Loss(reduction='none') # + weight + mean
        self.augmentation = dict()
        for key, value in config.items():
            if 'AUG_' in key:
                self.augmentation[key] = value
        self.train_data = DatasetCAMUS(dataset_path='../../datasets/CAMUS/',
                                       input_name=config['INPUT_NAME'],
                                       target_name=config['TARGET_NAME'],
                                       condition_name=config['CONDITION_NAME'],
                                       img_res=config['IMAGE_RES'],
                                       target_rescale=config['TARGET_TRANS'],
                                       input_rescale=config['INPUT_TRANS'],
                                       condition_rescale=config['CONDITION_TRANS'],
                                       labels=config['LABELS'],
                                       train_ratio=0.95,
                                       valid_ratio=0.02,
                                       augment=self.augmentation,
                                       subset='train')
        self.valid_data = DatasetCAMUS(dataset_path='../../datasets/CAMUS/',
                                       input_name=config['INPUT_NAME'],
                                       target_name=config['TARGET_NAME'],
                                       condition_name=config['CONDITION_NAME'],
                                       img_res=config['IMAGE_RES'],
                                       target_rescale=config['TARGET_TRANS'],
                                       input_rescale=config['INPUT_TRANS'],
                                       condition_rescale=config['CONDITION_TRANS'],
                                       labels=config['LABELS'],
                                       train_ratio=0.95,
                                       valid_ratio=0.02,
                                       augment=self.augmentation,
                                       subset='valid')

        self.test_data = DatasetCAMUS(dataset_path='../../datasets/CAMUS/',
                                      input_name=config['INPUT_NAME'],
                                      target_name=config['TARGET_NAME'],
                                      condition_name=config['CONDITION_NAME'],
                                      img_res=config['IMAGE_RES'],
                                      target_rescale=config['TARGET_TRANS'],
                                      input_rescale=config['INPUT_TRANS'],
                                      condition_rescale=config['CONDITION_TRANS'],
                                      labels=config['LABELS'],
                                      train_ratio=0.95,
                                      valid_ratio=0.02,
                                      augment=self.augmentation,
                                      subset='test')
        self.train_loader = torch.utils.data.DataLoader(self.train_data,
                                                        batch_size=16, # 32
                                                        shuffle=True,
                                                        num_workers=4)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_data,
                                                        batch_size=16,
                                                        shuffle=False,
                                                        num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(self.test_data,
                                                       batch_size=8,
                                                       shuffle=False,
                                                       num_workers=4)

    def train(self):
        #start_time = datetime.datetime.now()
        #batch_size = self.batch_size
        #max_iter = self.max_iter
        #val_interval = self.val_interval
        #log_interval = self.log_interval
        #ave_model_interval = self.save_model_interval
        patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)

        prev_time = time.time()
        epochs = 100
        for epoch in range(epochs):
            for i, batch in enumerate(self.train_loader):
                target, condition, input_, weight_map_condition = batch

                condition = condition.to(self.device)
                real_echo = target.to(self.device)
                input_ = input_.to(self.device)
                weight_map_condition = weight_map_condition.to(self.device)

                # Adversarial ground truths

                valid = torch.tensor(np.ones((condition.size(0), *patch)), dtype=torch.float32, device=self.device)
                # valid = Variable(torch.from_numpy(np.ones((real_A.size(0), *patch)).astype(float)).to(device),
                # requires_grad=False) fake = Variable(torch.from_numpy(np.zeros((real_A.size(0), *patch)).astype(
                # float)).to(device), requires_grad=False)
                fake = torch.tensor(np.zeros((condition.size(0), *patch)), dtype=torch.float32, device=self.device)

                # ------------------
                #  Train Generators
                # ------------------

                self.optimizer_G.zero_grad()

                # GAN loss
                fake_echo = self.generator(condition)
                pred_fake = self.discriminator(fake_echo, condition)
                loss_GAN = self.criterion_GAN(pred_fake, fake)  # valid

                # Pixel-wise loss
                # loss_pixel = torch.mean(criterion_pixelwise(fake_echo, real_echo) * weight_map_condition)
                loss_pixel = self.criterion_pixelwise(fake_echo, real_echo)

                # Total loss
                loss_G = loss_GAN + loss_pixel  # * lambda_pixel

                loss_G.backward()

                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Real loss
                pred_real = self.discriminator(real_echo, condition)
                loss_real = self.criterion_GAN(pred_real, valid)

                # Fake loss
                pred_fake = self.discriminator(fake_echo.detach(), condition)
                loss_fake = self.criterion_GAN(pred_fake, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                self.optimizer_D.step()

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(self.train_loader) + i
                batches_left = epochs * len(self.train_loader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                    % (
                        epoch,
                        epochs,
                        i,
                        len(self.train_loader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_pixel.item(),
                        loss_GAN.item(),
                        time_left,
                    )
                )
                # If at sample interval save image
                if batches_done % 100 == 0:
                    self.sample_images(batches_done)
                self.step += 1
                if self.use_wandb:
                    import wandb
                    wandb.log({'loss_D': loss_D, 'loss_real_D': loss_real, 'loss_fake_D': loss_fake,
                               'loss_G': loss_G, 'loss_pixel': loss_pixel, 'loss_GAN': loss_GAN},

                              step=self.step)

    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def sample_images(self, batches_done):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(self.valid_loader))
        # real_A = Variable(imgs["B"].type(Tensor))
        # real_B = Variable(imgs["A"].type(Tensor))
        condition = imgs[0].to(self.device)
        real_echo = imgs[1].to(self.device)
        fake_echo = self.generator(condition)
        img_sample = torch.cat((condition.data, fake_echo.data, real_echo.data), -2)
        save_image(img_sample, "images/%s.png" % (batches_done), nrow=4, normalize=True)
