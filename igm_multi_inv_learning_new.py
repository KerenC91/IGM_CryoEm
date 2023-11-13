import os

# -*- coding: utf-8 -*-
"""IGM_learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19oShgRrqugnNau0WwKlGUsWKDDPKLL5T
"""

##from google.colab import drive
#drive.mount('/content/drive')

#cd 'drive/MyDrive/ML_projects/IGM-journal'
import matplotlib
import numpy as np
import torch
#import torch.nn as nn
#import torch.nn.init as init
#import torch.nn.functional as functional
#from torchvision import datasets, transforms
#from torchvision.utils import save_image
#import matplotlib.pyplot as plt
#import matplotlib
#from torch.autograd import Variable

#torch.set_default_dtype(torch.float32)
#import torch.optim as optim
#import pickle
#import math
#from torch import Tensor
import json

## IMPORTS
#import generative_model
#import utils
import generative_model.model_utils as model_utils
import utils.training_utils as training_utils
import utils.data_utils as data_utils
import debug.debug as debug
import torch.distributed as dist
from torch.utils.data import TensorDataset

# from sys import exit
#import matplotlib.pyplot as plt
#from torch.nn import functional as F
import random
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
#torch.cuda.set_device(2)
#torch.cuda.empty_cache()


DEBUG = False

class MyDataset(Dataset):
    
    def __init__(self, tensor_list, noisy_targets):
        self.tensor_list = tensor_list
        self.noisy_targets = noisy_targets

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        mu = self.tensor_list[idx][0].unsqueeze(0) #torch.Size([1, 40])
        L = self.tensor_list[idx][1] #torch.Size([40, 40])
        target = self.noisy_targets[idx]
        
        mu = mu.to('cpu')
        L = L.to('cpu')
        target = target.to('cpu')
        return torch.cat([mu, L], dim=0), target

def ddp_setup():
    #os.environ["MASTER_ADDR"] = "localhost"
    #os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl")#, rank=0, world_size=4)
    #torch.cuda.set_device(0)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def main_function(args):

    ddp_setup()
    # Get noisy + true data
    if 'multi' in args.task and 'compressed-sensing' in args.task:
        sigmas = [args.sigma_den, args.sigma_cs]
    elif 'multi' in args.task and 'phase-retrieval' in args.task:
        sigmas = [args.sigma_den, args.sigma_pr]
    elif 'closure-phase' in args.task:
        sigmas = [args.sigma, args.sigma_cp]
    else:
        sigmas = args.sigma
     
    true, noisy, A, sigma, kernels = data_utils.get_true_and_noisy_data(image_size=args.image_size,
                                 sigma=sigmas,
                                 num_imgs_total=args.num_imgs,
                                 dataset=args.dataset,
                                 class_idx=args.class_idx,
                                 task=args.task,
                                 objective=args.objective,
                                 front_facing=args.front_facing,
                                 cphase_count=args.cphase_count,
                                 envelope_params=args.envelope,
                                 rand_shift=args.rand_shift)
    
    # Get generator:
    # - generator = neural network params
    # - G = functional form
    generator, G = model_utils.get_generator(args.latent_dim, args.image_size, args.generator_type)
    print("Generator number of parameters: {0}".format(model_utils.count_params(generator)))

    # Get latent GMM model
    models = model_utils.get_latent_model(args.latent_dim, args.num_imgs, args.latent_type)
    #I have to convert models to be a Dataset, as in from torch.utils.data import Dataset
    dataset = MyDataset(models, noisy)
    train_data = DataLoader(
        dataset, # training_data
        batch_size=1,#args.num_imgs,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
    # for mu, L in train_data: 
    #     mu = mu.to(device)
    #     L = L.to(device)
    #     print(f"{mu.shape}, {L.shape}")

    trainer_args = training_utils.init_trainer(model=generator,
                                               train_data=train_data,
                                               save_every=args.save_every,
                                               snapshot_path=f'./{args.sup_folder}/{args.folder}/snapshot.pt')
    # Learn the IGM
    models, generator = training_utils.train_latent_gmm_and_generator(models=models,
                                          generator=generator,
                                          generator_func=G,
                                          generator_type=args.generator_type,
                                          lr=args.lr,
                                          sigma=sigma,
                                          targets=noisy,
                                          true_imgs=true,
                                          num_samples=args.num_samples,
                                          num_imgs_show=args.num_imgs_show,
                                          num_imgs=args.num_imgs,
                                          num_epochs=args.num_epochs,
                                          As=A,
                                          task=args.task,
                                          save_img=args.save_img,
                                          dropout_val=args.dropout_val,
                                          layer_size=args.layer_size,
                                          num_layer_decoder=args.num_layer_decoder,
                                          batchGD=args.batchGD,
                                          dataset=args.dataset,
                                          class_idx=args.class_idx,
                                          seed=args.seed,
                                          GMM_EPS=args.GMM_EPS,
                                          sup_folder=args.sup_folder,
                                          folder=args.folder,
                                          latent_model=args.latent_type,
                                          image_size=args.image_size,
                                          num_channels=args.num_channels,
                                          no_entropy=args.no_entropy, 
                                          eps_fixed = args.eps_fixed,
                                          gamma=args.gamma,
                                          cphase_anneal=args.cphase_anneal,
                                          cphase_scale=args.cphase_scale,
                                          envelope_params=args.envelope,
                                          centroid_params=args.centroid,
                                          locshift_params=args.locshift,
                                          from_checkpoint=args.from_checkpoint,
                                          validate_args=args.validate_args,
                                          normalize_loss=args.normalize_loss,
                                          trainer_args=trainer_args)
    destroy_process_group()




if __name__ == "__main__":

    ################################################## SETUP ARGUMENTS ########################################################

    parser = argparse.ArgumentParser(description='Learning the IGM')

    # sigmas
    parser.add_argument('--sigma', type=float, default=None, metavar='N',
            help='noise std (default: None)')
    parser.add_argument('--sigma_den', type=float, default=0.70710678118, metavar='N',
            help='noise std (default: sqrt(0.5))')
    parser.add_argument('--sigma_cs', type=float, default=0.31622776601, metavar='N',
            help='noise std (default: sqrt(0.1))')
    parser.add_argument('--sigma_pr', type=float, default=0.31622776601, metavar='N',
            help='noise std (default: sqrt(0.1))')
    parser.add_argument('--sigma_cp', type=float, default=None, metavar='N',
            help='closure phase noise std (default: None)')

    # optimization params
    parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
            help='learning rate (default: 1e-4)')
    parser.add_argument('--num_samples', type=int, default=12, metavar='N',
            help='number of samples drawn from generator for batch size (default: 12)')
    parser.add_argument('--num_epochs', type=int, default=100000, metavar='N',
            help='number of epochs to train (default: 100000)')
    
    # dataset setup
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
            help='dataset to use (default: MNIST)')
    parser.add_argument('--front_facing', action='store_true', default=True,
            help='choosing only images from face dataset with faces facing forward (default: True )')
    parser.add_argument('--class_idx', type=int, default=8, metavar='N',
            help='class index for MNIST (default: 8)')
    parser.add_argument('--num_imgs', type=int, default=75, metavar='N',
            help='number of total examples (default: 75)')
    parser.add_argument('--num_imgs_show', type=int, default=5, metavar='N',
            help='number of examples to plot (default: 5)')
    parser.add_argument('--num_channels', type=int, default=1, metavar='N',
            help='either 1 or 3 (grayscale or RGB) (default: 1)')
    parser.add_argument('--image_size', type=int, default=32, metavar='N',
            help='size of image (default: 32)')
    
    # IGM architecture params
    parser.add_argument('--generator_type', type=str, default='deepdecoder', metavar='N',
            help='generator architecture to use (default: deepdecoder)')
    parser.add_argument('--latent_type', type=str, default='gmm', metavar='N',
            help='model for latent variational distribution (default: gmm)')
    parser.add_argument('--eps_fixed', action='store_true', default=False,
            help='keep epsilon fixed for gmm low rank (default: False)')
    parser.add_argument('--num_layer_decoder', type=int, default=6, metavar='N',
            help='number of layers in deep decoder (default: 6)')
    parser.add_argument('--layer_size', type=int, default=150, metavar='N',
            help='size of hidden layers of deep decoder (default: 150)')
    parser.add_argument('--dropout_val', type=float, default=1e-4, metavar='N',
            help='amount of dropout used in generator (default: 1e-4)')
    parser.add_argument('--GMM_EPS', type=float, default=1e-3, metavar='N',
            help='amount of perturbation added to gaussian latent distributions (default: 1e-3)')
    parser.add_argument('--latent_dim', type=int, default=50, metavar='N',
            help='size of latent dimension of generator (default: 50)')
    
    # problem setup
    parser.add_argument('--task', type=str, default='denoising', metavar='N',
            help='inverse problem to solve (default: denoising)')
    parser.add_argument('--objective', type=str, default='learning', metavar='N',
            help='either learn prior or perform model selection (default: learning)')

    # closure phase problem params
    parser.add_argument('--gamma', type=float, default=0.001, metavar='N',
                        help='cphase loss = gamma * loss_mag + len(meas_mag) / len(meas_phase) * loss_phase '
                             '(default: 0.001)')
    parser.add_argument('--cphase_count', type=str, default='min-cut0bl',
                        help='number of closure phases ("min", "max", or "min-cut0bl", '
                             'default: "min-cut0bl")')
    parser.add_argument('--cphase_anneal', type=int, default=None, metavar='N',
                        help='if >0, epoch at which to begin annealing the phase portion of the loss '
                             '(default: None)')
    parser.add_argument('--cphase_scale', type=float, default=1, metavar='N',
                        help='multiply the phase portion of loss by this amount '
                             '(default: 1)')
    parser.add_argument('--envelope', nargs=3, default=None, metavar=('etype', 'ds1', 'ds2'),
                        help='type (either "sq" or "circ") and params of envelope to encourage '
                             'centering, if desired (default: None)')
    parser.add_argument('--centroid', nargs='*', default=None, metavar=('weight', 'anneal'),
                        help='weight on centroid loss term to encourage centering, if desired, '
                             'and epoch at which to start annealing this term (default: None)')
    parser.add_argument('--locshift', nargs=4, default=None, metavar=('etype', 'ds1', 'ds2', 'learn'),
                        help='generate centralized image and shift location with convolutions with '
                             'given envelope params (default: None)')

    # other params
    parser.add_argument('--seed', type=int, default=100, metavar='N',
            help='random seed (default: 100)')
    parser.add_argument('--save_img', action='store_true', default=True,
            help='whether or not to save while training (default: True )')
    
    parser.add_argument('--batchGD', action='store_true', default=False,
            help='use batch gradient descent (default: False )')
    parser.add_argument('--no_entropy', action='store_true', default=False,
            help='no entropy for loss (default: False )')
    parser.add_argument('--from_checkpoint', action='store_true', default=False,
                        help='continue training from latest checkpoint, instead of from epoch 0'
                             '(default: False)')
    parser.add_argument('--validate_args', action='store_true', default=False,
                        help='value of validate_args for latent distribution in training loop '
                             '(default: False)')
    parser.add_argument('--suffix', type=str, default='',
                        help='string add to end of directory name for hacky reasons '
                             '(default: "")')

    parser.add_argument('--normalize_loss', action='store_true', default=False,
                        help='normalizing the losses.'
                             '(default: False)')
    parser.add_argument('--rand_shift', action='store_true', default=False,
                        help='perform shifts on a single image, perform learning o those augmentations'
                             '(default: False)')
    #parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot')
    #parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')

    args = parser.parse_args()
    ## debugging args - Keren
    if DEBUG is True:
        debug.set_args(args)
    ## debugging end
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = int(os.environ["LOCAL_RANK"])
    #device = 'cpu'
    
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #random.seed(args.seed)

    #GPU = torch.cuda.is_available()
    #if GPU == True:
        #torch.backends.cudnn.enabled = True
        #torch.backends.cudnn.benchmark = True
        #dtype = torch.cuda.FloatTensor
    #    dtype = torch.FloatTensor
    #    print("num GPUs",torch.cuda.device_count())
    #else:
    #    dtype = torch.FloatTensor
    dtype = torch.FloatTensor
    
    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

    # Process parameters
    if args.envelope:
        assert args.envelope[0] == 'sq' or args.envelope[0] == 'circ'
        args.envelope[1] = int(args.envelope[1])
        args.envelope[2] = int(args.envelope[2])
    if args.centroid:
        assert len(args.centroid) == 1 or len(args.centroid) == 2
        args.centroid[0] = float(args.centroid[0])
        if len(args.centroid) == 1:
            args.centroid.append(None)
        else:
            args.centroid[1] = int(float(args.centroid[1]))
    if args.locshift:
        assert args.locshift[0] == 'sq' or args.locshift[0] == 'circ'
        args.locshift[1] = int(args.locshift[1])
        args.locshift[2] = int(args.locshift[2])
        assert args.locshift[3] == 'l' or args.locshift[3] == 'nl'

     ## Saving parameters
    if args.batchGD == True:
        args.sup_folder = "results_batched"
    else:
        args.sup_folder = "results_SGD"
    if "multi" in args.task:
        args.sup_folder += f"_multi_{args.sigma_cs}"
    if args.dataset == "MNIST" and args.class_idx is not None: 
        args.folder = f"{args.dataset}{args.image_size}{args.class_idx}_{args.latent_type}_{args.task}_{args.generator_type}_{str(args.num_imgs)}imgs_{str(args.sigma)}noise_std_dropout{args.dropout_val}_layer_size{args.layer_size}x{args.num_layer_decoder}_latent{args.latent_dim}_seed{args.seed}"
    else:
        args.folder = f"{args.dataset}{args.image_size}_{args.latent_type}_{args.task}_{args.generator_type}_{str(args.num_imgs)}imgs_{str(args.sigma)}noise_std_dropout{args.dropout_val}_layer_size{args.layer_size}x{args.num_layer_decoder}_latent{args.latent_dim}_seed{args.seed}"
    if args.latent_type == "gmm" or args.latent_type == "gmm_eye" or args.latent_type == "gmm_custom" or args.latent_type == "gmm_low_eye" or args.latent_type == "gmm_low":
        args.folder += f"_eps{args.GMM_EPS}"
    if args.latent_type == "gmm_low_eye" or args.latent_type == "gmm_low":
        args.folder += f"_{args.eps_fixed}"
    if args.no_entropy==True:
        args.folder += "_no_entropy"
    if "closure-phase" in args.task:
        args.folder += f"_gamma{args.gamma}"
        args.folder += f"_sigma-cp{args.sigma_cp}"
        args.folder += f"_cphases-{args.cphase_count}"
        if args.cphase_anneal is not None and args.cphase_anneal > 0:
            args.folder += f"_cp-anneal{args.cphase_anneal}"
        if args.cphase_scale != 1:
                args.folder += f"_cp-scale{args.cphase_scale}"
        if args.envelope:
            args.folder += f"_envelope{args.envelope[0]}-{args.envelope[1]}-{args.envelope[2]}"
        if args.centroid:
            args.folder += f"_centroid{args.centroid[0]:.0e}-{args.centroid[1]:.0e}"
        if args.locshift:
            args.folder += f"_locshift{args.locshift[0]}-{args.locshift[1]}-{args.locshift[2]}-{args.locshift[3]}"

    if args.suffix != '':
        args.folder += f"_{args.suffix}"

    if not os.path.exists(f'./{args.sup_folder}/{args.folder}'):
        os.makedirs(f'./{args.sup_folder}/{args.folder}')
    if not os.path.exists(f'./{args.sup_folder}/{args.folder}/model_checkpoints'):
        os.makedirs(f'./{args.sup_folder}/{args.folder}/model_checkpoints')
    
    with open("{}/args.json".format(f'./{args.sup_folder}/{args.folder}'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    if args.sigma is None and args.dataset in ("m87", "sagA_video"):
        if args.dataset=="sagA_video":
            print("load matrix of sigmas")
            sigma = np.load("./utils/sigma_ngEHT_sagA_video_.5x.npz", allow_pickle=True)
        elif args.dataset == "m87":
            print("load matrix of sigmas")
            sigma = np.load("./utils/sigma_EHT2025_m87_frame.npz", allow_pickle=True)
        
        args.sigma = []
        for i in range(args.num_imgs):
            args.sigma.append(torch.tensor(sigma['arr_0'][i][np.newaxis, :, np.newaxis]).to(device))

    main_function(args)
