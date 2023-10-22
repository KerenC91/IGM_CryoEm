import numpy as np
Namespace = dict


def set_args(args: Namespace):
    args.task = 'denoising'
    args.latent_dim = 40
    args.generator_type = 'deepdecoder'
    args.num_imgs = 5
    args.dataset = 'MNIST'
    args.sigma = 0.5
    args.image_size = 64
    args.batchGD = True
    args.class_idx = 8
    args.GMM_EPS = 1e-3
    args.latent_type = "gmm"


def get_loss(f_loss, f_loss_ent, f_loss_prior, f_loss_data, num_epoch):

    try:
        # Read npy files
        data_loss = np.load(f_loss)
        data_loss_ent = np.load(f_loss_ent)
        data_loss_prior = np.load(f_loss_prior)
        data_loss_data = np.load(f_loss_data)
    except FileNotFoundError:
        print("Could not read one of the following files:")
        print(f_loss)
        print(f_loss_ent)
        print(f_loss_prior)
        print(f_loss_data)

    # Get loss data in epoch num_epoch
    print("epoch ", num_epoch, ": data_loss=", data_loss[num_epoch], ", data_loss_ent=", data_loss_ent[num_epoch],
          ", data_loss_prior=", data_loss_prior[num_epoch], ", data_loss_data=", data_loss_data[num_epoch])
    return (data_loss[num_epoch], data_loss_ent[num_epoch],
            data_loss_prior[num_epoch], data_loss_data[num_epoch])
