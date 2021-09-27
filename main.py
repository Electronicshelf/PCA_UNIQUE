import os
import argparse
# from solver_9 import Solver
from data_loader import get_loader
from torch.backends import cudnn
import os, ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # # Create directories if not exist.
    # if not os.path.exists(config.log_dir):
    #     os.makedirs(config.log_dir)
    # if not os.path.exists(config.model_save_dir):
    #     os.makedirs(config.model_save_dir)
    # if not os.path.exists(config.sample_dir):
    #     os.makedirs(config.sample_dir)
    # if not os.path.exists(config.result_dir):
    #     os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    celeba_loader_skt = None
    rafd_loader = None
    dataX = None
    dataY = None

    if config.dataset in ["Fair_Face"]:
        # print("try")
        celeba_loader, dataX = get_loader(config.image_dir_125, config.attr_path_train, config.attr_path_val, config.selected_attrs,
                                          config.ff_crop_size, config.image_size, config.batch_size,
                                          'Fair_Face', config.mode, config.num_workers)

        celeba_loader_skt, dataY = get_loader(config.image_dir_025, config.attr_path_train, config.attr_path_val, config.selected_attrs,
                                              config.ff_crop_size, config.image_size, config.batch_size,
                                              'Fair_Face', config.mode, config.num_workers)


    # # Solver for training and testing StarGAN.
    # solver = Solver(celeba_loader, celeba_loader_skt, rafd_loader, config, dataX, dataY)
    #
    # if config.mode == 'train':
    #     if config.dataset in ['CelebA', 'RaFD']:
    #         solver.train()
    #     elif config.dataset in ['Both']:
    #         solver.train_multi()
    # elif config.mode == 'test':
    #     if config.dataset in ['CelebA', 'RaFD']:
    #         solver.test()
    #     elif config.dataset in ['Both']:
    #         solver.test_multi()
    #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=5, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--ff_crop_size', type=int, default=224, help='crop size for the Fair Face dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=100, help='weight for reconstruction loss  [10]')
    parser.add_argument('--lambda_gp', type=float, default=5, help='weight for gradient penalty [10]')

    # Training configuration.
    parser.add_argument('--dataset', type=str, default='Fair_Face', choices=['Fair_Face'])
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=10000000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.000002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.00002, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=10, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the Fair Face dataset',
                        default=['Indian'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=150000, help='test model from this step')
    # 3800000 3900000 360000 3880000
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories. /media/sdb2/celeba/resized_celebA
    # parser.add_argument('--celeba_image_dir', type=str, default='/media/sdb2/celeba/resized_celebA/celebA')
    parser.add_argument('--image_dir_125', type=str,
                        default='/media/electronicshelf/358251B7513FAF65/Dataset/fairface-img-margin125-trainval')
    parser.add_argument('--image_dir_025', type=str,
                        default='/media/electronicshelf/358251B7513FAF65/Dataset/fairface-img-margin025-trainval')
    parser.add_argument('--attr_path_train', type=str,
                        default='/media/electronicshelf/358251B7513FAF65/Dataset/fairface_label_train.csv')
    parser.add_argument('--attr_path_val', type=str,
                        default='/media/electronicshelf/358251B7513FAF65/Dataset/fairface_label_val.csv')
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)  # 1000
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
