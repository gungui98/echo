import numpy as np
import torch
from models import GeneratorUNet, Discriminator
from data_loader_camus import DatasetCAMUS
from torchvision.utils import save_image

# from torchsummary import summary
import datetime
import time
import sys
import random
import os

RESULT_DIR = 'results'
VAL_DIR = 'val_images'
TEST_DIR = 'test_images'
MODELS_DIR = 'saved_models'

SEED = 17


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


class GAN:
    def __init__(self, config, use_wandb, device, dataset_path):

        # Configure data loader
        self.config = config
        self.result_name = config['NAME']

        self.use_wandb = use_wandb
        self.device = device
        self.epochs = config['EPOCHS']
        self.log_interval = config['LOG_INTERVAL']
        self.step = 0
        self.loaded_epoch = 0
        self.epoch = 0
        self.base_dir = './'

        self.patch = (1, config['PATCH_SIZE'], config['PATCH_SIZE'])
        # self.patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)

        # Input shape
        self.channels = config['CHANNELS']
        self.img_rows = config['IMAGE_RES'][0]
        self.img_cols = config['IMAGE_RES'][1]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        assert self.img_rows == self.img_cols, 'The current code only works with same values for img_rows and img_cols'

        # Input images and their conditioning images

        self.conditional_d = config.get('CONDITIONAL_DISCRIMINATOR', False)
        self.recon_loss = config.get('RECON_LOSS', 'basic')
        self.loss_weight_d = config["LOSS_WEIGHT_DISC"]
        self.loss_weight_g = config["LOSS_WEIGHT_GEN"]

        # Calculate output shape of D (PatchGAN)
        patch_size = config['PATCH_SIZE']
        patch_per_dim = int(self.img_rows / patch_size)
        self.num_patches = (patch_per_dim, patch_per_dim, 1)

        # Number of filters in the first layer of G and D
        self.gf = config['FIRST_LAYERS_FILTERS']
        self.df = config['FIRST_LAYERS_FILTERS']
        self.skipconnections_generator = config['SKIP_CONNECTIONS_GENERATOR']
        self.output_activation = config['GEN_OUTPUT_ACT']
        self.decay_factor_G = config['LR_EXP_DECAY_FACTOR_G']
        self.decay_factor_D = config['LR_EXP_DECAY_FACTOR_D']

        self.generator = GeneratorUNet(in_channels=self.channels, out_channels=self.channels).to(self.device)
        self.discriminator = Discriminator(in_channels=self.channels).to(self.device)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=config['LEARNING_RATE_G'],
                                            betas=(config['ADAM_B1'], 0.999))  # 0.0002
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=config['LEARNING_RATE_D'], betas=(config['ADAM_B1'], 0.999))

        self.criterion_GAN = torch.nn.MSELoss().to(self.device)

        self.criterion_pixelwise = torch.nn.L1Loss(reduction='none').to(self.device)  # MAE
        # criterion_pixelwise = torch.nn.L1Loss(reduction='none') # + weight + mean

        self.augmentation = dict()
        for key, value in config.items():
            if 'AUG_' in key:
                self.augmentation[key] = value

        self.train_data = DatasetCAMUS(dataset_path=dataset_path,
                                       random_state=config['RANDOM_SEED'],
                                       img_size=config['IMAGE_RES'],
                                       classes=config['LABELS'],
                                       train_ratio=config['TRAIN_RATIO'],
                                       valid_ratio=config['VALID_RATIO'],
                                       # augment=self.augmentation,
                                       subset='train')
        self.valid_data = DatasetCAMUS(dataset_path=dataset_path,
                                       random_state=config['RANDOM_SEED'],
                                       img_size=config['IMAGE_RES'],
                                       classes=config['LABELS'],
                                       train_ratio=config['TRAIN_RATIO'],
                                       valid_ratio=config['VALID_RATIO'],
                                       # augment=self.augmentation,
                                       subset='valid')

        self.test_data = DatasetCAMUS(dataset_path=dataset_path,
                                      random_state=config['RANDOM_SEED'],
                                      img_size=config['IMAGE_RES'],
                                      classes=config['LABELS'],
                                      train_ratio=config['TRAIN_RATIO'],
                                      valid_ratio=config['VALID_RATIO'],
                                      # augment=self.augmentation,
                                      subset='test')

        self.train_loader = torch.utils.data.DataLoader(self.train_data,
                                                        batch_size=config['BATCH_SIZE'],  # 32 max
                                                        shuffle=True,
                                                        num_workers=config['NUM_WORKERS'])
        self.valid_loader = torch.utils.data.DataLoader(self.valid_data,
                                                        batch_size=config['BATCH_SIZE'],
                                                        shuffle=False,
                                                        num_workers=config['NUM_WORKERS'])
        self.test_loader = torch.utils.data.DataLoader(self.test_data,
                                                       batch_size=config['BATCH_SIZE'],
                                                       shuffle=False,
                                                       num_workers=config['NUM_WORKERS'])

        # Training hyper-parameters
        self.batch_size = config['BATCH_SIZE']
        # self.max_iter = config['MAX_ITER']
        self.val_interval = config['VAL_INTERVAL']
        self.log_interval = config['LOG_INTERVAL']
        self.save_model_interval = config['SAVE_MODEL_INTERVAL']
        self.lr_G = config['LEARNING_RATE_G']
        self.lr_D = config['LEARNING_RATE_D']

    def train(self):

        prev_time = time.time()
        batch_size = self.batch_size
        # max_iter = self.max_iter
        val_interval = self.val_interval
        log_interval = self.log_interval
        save_model_interval = self.save_model_interval

        # Adversarial ground truths for discriminator losses

        # valid = torch.tensor(np.ones((batch_size,) + self.num_patches), dtype=torch.float32, device=self.device)
        # fake = torch.tensor(np.zeros((batch_size,) + self.num_patches), dtype=torch.float32, device=self.device)

        for epoch in range(self.loaded_epoch, self.epochs):
            self.epoch = epoch
            for i, batch in enumerate(self.train_loader):

                self.generator.train()
                self.discriminator.train()

                target, target_gt, inputs, weight_map, quality, heart_state, view = batch

                target_gt = target_gt.to(self.device)
                target = target.to(self.device)
                inputs = inputs.to(self.device)  # not used
                weight_map = weight_map.to(self.device)  # not used

                valid = torch.tensor(np.ones((target_gt.size(0), *self.patch)), dtype=torch.float32, device=self.device)
                fake = torch.tensor(np.zeros((target_gt.size(0), *self.patch)), dtype=torch.float32, device=self.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()
                fake_echo = self.generator(target_gt)

                # if self.conditional_d:
                # Real loss
                pred_real = self.discriminator(target, target_gt)
                loss_real = self.criterion_GAN(pred_real, valid)

                # Fake loss
                pred_fake = self.discriminator(fake_echo.detach(), target_gt)
                loss_fake = self.criterion_GAN(pred_fake, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                self.optimizer_D.step()

                # ------------------
                #  Train Generators
                # ------------------

                self.optimizer_G.zero_grad()

                # GAN loss
                fake_echo = self.generator(inputs)
                pred_fake = self.discriminator(fake_echo, target_gt)
                loss_GAN = self.criterion_GAN(pred_fake, fake)  # valid

                # Pixel-wise loss
                loss_pixel = torch.mean(self.criterion_pixelwise(fake_echo, target) * weight_map)
                # loss_pixel = self.criterion_pixelwise(fake_echo, target)

                # Total loss
                loss_G = self.loss_weight_d * loss_GAN + self.loss_weight_g * loss_pixel  # 100
                # loss_G = loss_GAN + loss_pixel

                loss_G.backward()

                self.optimizer_G.step()

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = self.epoch * len(self.train_loader) + i
                batches_left = self.epochs * len(self.train_loader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f fake: %f real: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                    % (
                        self.epoch,
                        self.epochs,
                        i,
                        len(self.train_loader),
                        loss_D.item(),
                        loss_fake.item(),
                        loss_real.item(),
                        loss_G.item(),
                        loss_pixel.item(),
                        loss_GAN.item(),
                        time_left,
                    )
                )
                # save valid images
                self.generator.eval()
                self.discriminator.eval()

                if batches_done % self.log_interval == 0:
                    self.sample_images(batches_done)
                    self.sample_images2(batches_done)


                # log wandb
                self.step += 1
                if self.use_wandb:
                    import wandb
                    wandb.log({'loss_D': loss_D, 'loss_real_D': loss_real, 'loss_fake_D': loss_fake,
                               'loss_G': loss_G, 'loss_pixel': loss_pixel, 'loss_GAN': loss_GAN},

                              step=self.step)
            if epoch % save_model_interval == 0:
                self.save(f'{self.base_dir}/generator_last_checkpoint.bin', model='generator')
                self.save(f'{self.base_dir}/discriminator_last_checkpoint.bin', model='discriminator')

    def sample_images(self, batches_done):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(self.valid_loader))
        condition = imgs[0].to(self.device)
        real_echo = imgs[1].to(self.device)
        fake_echo = self.generator(condition)
        img_sample = torch.cat((condition.data, fake_echo.data, real_echo.data), -2)
        save_image(img_sample, "images/%s.png" % batches_done, nrow=4, normalize=True)

        #if self.use_wandb:
        #    import wandb
        #    wandb.log({'val_image': img_sample.cpu()}, step=self.step)

    def sample_images2(self, batches_done):
        """Saves a generated sample from the validation set"""
        target, target_gt, inputs, weight_map, quality, heart_state, view = next(iter(self.valid_loader))
        #
        target_gt = target_gt.to(self.device)
        target = target.to(self.device)
        quality = quality.to(self.device)
        fake_echo = self.generator(target)#, quality)
        # img_sample = torch.cat((target.data, fake_echo.data, target_gt.data), -2)
        target = target.cpu().detach().numpy()
        fake_echo = fake_echo.cpu().detach().numpy()
        target_gt = target_gt.cpu().detach().numpy()
        quality = quality.cpu().detach().numpy()

        batch = 5

        img_sample = np.concatenate([target,
                                     fake_echo,
                                     target_gt], axis=1)
        q = ['low', 'med', 'high']
        import matplotlib.pyplot as plt
        rows, cols = 3, batch
        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(rows, cols)
        cnt = 0
        for row in range(rows):
            for col in range(cols):
                class_label = np.argmax(quality[col], axis=1)[0]

                axs[row, col].imshow(img_sample[col, row, :, :], cmap='gray')
                axs[row, col].set_title(titles[row] + ' ' + q[class_label], fontdict={'fontsize': 6})
                axs[row, col].axis('off')
                cnt += 1

        # fig.savefig('%s/%s/%s/%s_%d.png' % (RESULT_DIR, self.result_name, VAL_DIR, prefix, step_num))
        fig.savefig("images/_%s.png" % batches_done)

        if self.use_wandb:
            import wandb
            wandb.log({'val_image': fig}, step=self.step)

    def save(self, path, model='generator'):
        if model == 'generator':
            self.generator.eval()
            torch.save({
                'model_state_dict': self.generator.state_dict(),
                'optimizer_state_dict': self.optimizer_G.state_dict(),

                # 'scheduler_state_dict': self.scheduler.state_dict(),
                # 'best_summary_loss': self.best_summary_loss,
                'epoch': self.epoch,
            }, path)
            print('\ngenerator saved, epoch ', self.epoch)
        elif model == 'discriminator':

            self.discriminator.eval()
            torch.save({
                'model_state_dict': self.discriminator.state_dict(),
                'optimizer_state_dict': self.optimizer_D.state_dict(),

                # 'optimizer_state_dict': self.optimizer.state_dict(),
                # 'scheduler_state_dict': self.scheduler.state_dict(),
                # 'best_summary_loss': self.best_summary_loss,
                'epoch': self.epoch,
            }, path)
            print('discriminator saved, epoch ', self.epoch)


    def load(self, path, model='generator'):
        if model == 'generator':
            checkpoint = torch.load(path)
            self.generator.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # self.best_summary_loss = checkpoint['best_summary_loss']
            self.loaded_epoch = checkpoint['epoch'] + 1
            print('generator loaded, epoch ', self.loaded_epoch)
        elif model == 'discriminator':
            checkpoint = torch.load(path)
            self.discriminator.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loaded_epoch = checkpoint['epoch'] + 1
            print('discriminator loaded, epoch ', self.loaded_epoch)

            # self.best_summary_loss = checkpoint['best_summary_loss']

