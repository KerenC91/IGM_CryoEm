import json

import numpy as np
import torch
import aspire
import matplotlib.pyplot as plt
import os
import yaml
from collections import OrderedDict
import matplotlib.patches as mpatches
import random

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
    args.normalize_loss = True
    args.suffix = "debug"
    args.nproc = 1
    args.batch_size = 1


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

def generate_random_color():
    return tuple(random.random() for _ in range(4))

def plot_losses_comparison(*args, **kwargs):
    '''
    args - arguments containing list of .npy loss file paths
    kwargs -

    compare loss data from different run sessions
    '''

    # Get the number of total args and kwargs
    args_list = args[0]
    num_args = len(args_list)
    num_kwargs = len(kwargs)

    losses_list = []
    i = 0
    jump_size = kwargs.get('jump_size', None)
    # Create lambda function taking values in jumps of jump_size
    jump_over_array = lambda array: array[::jump_size]
    base_folder = kwargs.get('base_folder', None)
    loss_name = kwargs.get('loss_name', None)
    loss_file = f"{loss_name}.npy"
    clip_epochs = kwargs.get('clip_epochs', None)
    # Get minimal number of epochs
    min_epoch = 99001
    if clip_epochs:
        for arg in args_list:
            # Load loss data per session
            path = os.path.join(base_folder, arg, loss_file)
            data_loss = np.load(path)
            min_epoch = min(len(data_loss), 100000) # 100000 is the default value
    # Loop over sessions
    for arg in args_list:
        # Load loss data per session
        path = os.path.join(base_folder, arg, loss_file)
        data_loss = np.load(path)
        data_loss = data_loss[:min_epoch-1]
        # Get jumps array
        data_loss_jumps = jump_over_array(data_loss)
        # Append to list of losses to plot
        losses_list.append(data_loss_jumps)
        i += 1

    output_folder = kwargs.get('output_folder', None)
    output_file = kwargs.get('output_file', None)

    title = kwargs.get('diff', "No diff found")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.figure()
    comparison_details = kwargs.get('comparison_details', None)
    for i in range(num_args):
        plt.plot(losses_list[i], label=comparison_details[i])
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel(f"{loss_name}")
    plt.title(f'{title}, epochs={min_epoch}')
    plt.savefig(f'./{output_folder}/{output_file}.png')
    plt.close()


def get_folders_list(comparison_param, comparison_details):
    folders = []

    if comparison_param == "num_imgs":
        latent_type = "gmm"
        sigma = 0.5
        latent_dim = 40
        folders = [
            f"MNIST648_{latent_type}_denoising_deepdecoder_{comp_param}imgs_{sigma}noise_std_dropout0.0001_layer_size150x6_latent{latent_dim}_seed100_eps0.001"
            for comp_param in comparison_details
        ]
    elif comparison_param == "latent_dim":
        latent_type = "gmm"
        num_imgs = 5
        sigma = 0.5
        folders = [
            f"MNIST648_{latent_type}_denoising_deepdecoder_{num_imgs}imgs_{sigma}noise_std_dropout0.0001_layer_size150x6_latent{comp_param}_seed100_eps0.001"
            for comp_param in comparison_details
        ]
    elif comparison_param == "sigma":
        latent_type = "gmm"
        num_imgs = 5
        latent_dim = 40
        folders = [
            f"MNIST648_{latent_type}_denoising_deepdecoder_{num_imgs}imgs_{comp_param}noise_std_dropout0.0001_layer_size150x6_latent{latent_dim}_seed100_eps0.001"
            for comp_param in comparison_details
        ]
    elif comparison_param == "latent_type":
        num_imgs = 5
        sigma = 1.0
        latent_dim = 40
        folders = [
            f"MNIST648_{comp_param}_denoising_deepdecoder_{num_imgs}imgs_{sigma}noise_std_dropout0.0001_layer_size150x6_latent{latent_dim}_seed100_eps0.001"
            for comp_param in comparison_details
        ]
    else:
        raise ValueError(f"Undefined comparison param: {comparison_param}")
    return folders


def write_comparison_args(output_folder, comparison_param, comparison_details, folders):
    labels = OrderedDict([
        ("comparison_param", comparison_param),
        ("comparison_details", comparison_details),
        ("folders", folders),
    ])
    with open(f'./{output_folder}/params_comp_{comparison_param}.yaml', "w") as f:
        yaml.dump(labels, f)

def parse_config_file(file):
    args = {}

    # Open the JSON file
    with open(file, "r") as f:
        json_string = f.read()

    # Deserialize the JSON string to a dictionary
    args = json.loads(json_string)
    return args
# def test_model(model_pt, data_mrc):
#
#     # Load the PyTorch model file
#     model = torch.load(model_pt)
#     data_mat = aspire.ReadMRC(data_mrc)
#     # Use the model to make predictions
#     predictions = model(data_mat)
#     return predictions