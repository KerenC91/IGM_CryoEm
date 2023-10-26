import numpy as np
import torch
import aspire
import matplotlib.pyplot as plt
import yaml
import os

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

def plot_losses_comparison(*args, **kwargs):
    '''
    args - arguments containing list of .npy loss file paths
    kwargs -

    compare loss data from different run sessions
    '''

    # Get the number of total args and kwargs
    num_args = len(args)
    num_kwargs = len(kwargs)

    losses_list = []
    i = 0
    jump_size = 100
    # Create lambda function taking values in jumps of jump_size
    jump_over_array = lambda array: array[::jump_size]
    # Loop over sessions
    for arg in args:
        # Load loss data per session
        data_loss = np.load(arg)
        # Get jumps array
        data_loss_jumps = jump_over_array(data_loss)
        # Append to list of losses to plot
        losses_list.append(data_loss_jumps)
        i += 1

    output_folder = kwargs.get('output_folder', None)
    title = kwargs.get('diff', "No diff found")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.figure()
    for i in range(num_args):
        plt.plot(losses_list[i], label=i)
    plt.legend()
    plt.yscale('log')
    plt.title(f'{title}')
    plt.savefig(f'./{output_folder}/losses_comp.png')
    plt.close()

    # Write the dictionary to a YAML file
    with open(f'./{output_folder}/params_comp.yaml', "w") as f:
        yaml.dump(args, f)


# def test_model(model_pt, data_mrc):
#
#     # Load the PyTorch model file
#     model = torch.load(model_pt)
#     data_mat = aspire.ReadMRC(data_mrc)
#     # Use the model to make predictions
#     predictions = model(data_mat)
#     return predictions