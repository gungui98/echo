import json
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import torch

from data_loader_camus import DatasetCAMUS
from gan import GAN

# from utils import set_backend

flags.DEFINE_string('dataset_path', None, 'Path of the dataset.')
flags.DEFINE_boolean('test', False, 'Test model and generate outputs on the test set')
flags.DEFINE_string('config', None, 'Config file for training hyper-parameters.')
flags.DEFINE_boolean('use_wandb', False, 'Use wandb for logging')
flags.DEFINE_string('wandb_resume_id', None, 'Resume wandb process with the given id')
flags.DEFINE_string('ckpt_load', None, 'Path to load the model')
flags.DEFINE_float('train_ratio', 0.95,
                   'Ratio of training data used for training and the rest used for testing. Set this value to 1.0 if '
                   'the data in the test folder are to be used for testing.')
flags.DEFINE_float('valid_ratio', 0.02, 'Ratio of training data used for validation')
flags.mark_flag_as_required('dataset_path')
flags.mark_flag_as_required('config')

FLAGS = flags.FLAGS

plt.switch_backend('agg')


def main(argv):
    # Load configs from file

    config = json.load(open(FLAGS.config))
    # set_backend()

    # Set name
    #name = '{}_{}_'.format(config['INPUT_NAME'], config['TARGET_NAME'])
    #for l in config['LABELS']:
    #    name += str(l)
    #config['NAME'] += '_' + name




    if FLAGS.use_wandb:
        import wandb
        resume_wandb = True if FLAGS.wandb_resume_id is not None else False
        wandb.init(config=config, resume=resume_wandb, id=FLAGS.wandb_resume_id, project='EchoGen')

    # Initialize GAN
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if FLAGS.use_wandb:
        import wandb
        resume_wandb = True if FLAGS.wandb_resume_id is not None else False
        wandb.init(config=config, resume=resume_wandb, id=FLAGS.wandb_resume_id, project='EchoGen')

    model = GAN(config, FLAGS.use_wandb, device, FLAGS.dataset_path)


    # load trained models if they exist
    # if FLAGS.ckpt_load is not None:
    #    model.load_model(FLAGS.ckpt_load)

    if FLAGS.test:
        model.test()
    else:
        model.train()


if __name__ == '__main__':
    app.run(main)
