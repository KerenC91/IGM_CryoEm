# -*- coding: utf-8 -*-
"""IGM_learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19oShgRrqugnNau0WwKlGUsWKDDPKLL5T
"""
## IMPORTS
import os, sys, inspect
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import torch
import json
import generative_model.model_utils as model_utils
from utils.training_utils import Trainer
import utils.training_utils as training_utils
import utils.data_utils as data_utils
import debug.debug as debug
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
import time
import wandb
from datetime import datetime

# Globals
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
        return torch.cat([mu, L], dim=0), idx, target

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12532" #any free port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def get_sigmas(args):
    if 'multi' in args.task and 'compressed-sensing' in args.task:
        sigmas = [args.sigma_den, args.sigma_cs]
    elif 'multi' in args.task and 'phase-retrieval' in args.task:
        sigmas = [args.sigma_den, args.sigma_pr]
    elif 'closure-phase' in args.task:
        sigmas = [args.sigma, args.sigma_cp]
    else:
        sigmas = args.sigma
    
    return sigmas


def load_train_objs(rank, args):
    # Get noisy + true data
    sigmas = get_sigmas(args)
    
    true, noisy, A, sigma, kernels = data_utils.get_true_and_noisy_data(device=rank, image_size=args.image_size,
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
    # if rank == 0:
    #     print("true images:")
    #     for i in range(args.num_imgs):
    #         print(f'{i} {true[i]}')
            
    # Get generator:
    # - generator = neural network params
    # - G = functional form
    generator, G = model_utils.get_generator(rank, args.latent_dim, args.image_size, args.generator_type)
    print("Generator number of parameters: {0}".format(model_utils.count_params(generator)))

    # Get latent GMM model
    models = model_utils.get_latent_model(rank, args.latent_dim, args.num_imgs, args.latent_type)
    #I have to convert models to be a Dataset, as in from torch.utils.data import Dataset
    dataset = MyDataset(models, noisy)

    return true, noisy, A, sigma, kernels, models,\
        dataset, generator, G

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main_function(rank, args):
    # Apply ddp setup
    ddp_setup(rank, args.nproc)

    true, noisy, A, sigma, kernels, models,\
    dataset, generator, G = load_train_objs(rank, args)
    
    # if rank == 0:
    #     for i in range(args.num_imgs):
    #         print(f'{i} {true[i]}')
        
    # kernels parameter unused
    train_data = prepare_dataloader(dataset, args.batch_size)
    trainer = Trainer(generator=generator, train_data=train_data, gpu_id=rank,
                      snapshot_path=f'./{args.sup_folder}/{args.folder}/snapshot.pt',
                      sigma=sigma, targets=noisy, true_imgs=true, As=A,
                      models=models, generator_func=G,
                      args=args)
    # Get start time
    if trainer.gpu_id not in [-1, 0]:
        dist.barrier()
    # Only gpu 0 operating now...
    if trainer.gpu_id == 0:
        wandb.login()
        wandb.init(project='IGM_CryoEm',
                   name = f"{args.folder}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                   config=args)
        wandb.watch(generator, log_freq=100)
        print(f"Running {os.path.basename(__file__)}"
              f" with {args.nproc} gpus, "
              f"{args.num_epochs} total epochs, "
              f"{args.num_imgs} images, "
              f"{args.batch_size}={len(next(iter(train_data))[0])} batch size, "
              f"save checkpoint every {args.save_every} epochs")
        print(f"Normalization factor={(trainer.world_size * trainer.batch_size * len(trainer.train_data))}")
        print(f"len(trainer.models)={len(trainer.models)}")

        start_time = time.time()
        dist.barrier()
    # Learn the IGM
    trainer.train_latent_gmm_and_generator()
    # Get end time
    if trainer.gpu_id not in [-1, 0]:
         dist.barrier()

    if trainer.gpu_id == 0:
    # Only gpu 0 operating now...
        end_time = time.time()
        print(f"Time taken to train in {os.path.basename(__file__)}: {end_time - start_time} seconds")
        dist.barrier()
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
    # Added by Keren
    parser.add_argument('--normalize_loss', action='store_true', default=True,
                        help='normalizing the losses.'
                             '(default: False)')
    parser.add_argument('--rand_shift', action='store_true', default=False,
                        help='perform shifts on a single image, perform learning o those augmentations'
                             '(default: False)')
    parser.add_argument('--save_every', type=int, default=100, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=None,
                        type=int, help='Input batch size on each device (default: None)')
    parser.add_argument('--nproc', default=torch.cuda.device_count(),
                        type=int, help='nproc, default is the number of available gpus on the machine')
    parser.add_argument('--sigma_loss', type=float, default=None, 
            help='loss data regularization. Multiply original sigma buy that factor for'
            'loss data calulation.' 
            'sigma_loss > sigma (default: None)')
    parser.add_argument('--total_variation', type=float, default=None, 
            help='total variation weight. Add total variation regularization to the output image.' 
            ' (default: None)') 
    parser.add_argument('--wandb_log_interval', type=int, default=100,
            help='log interval over epochs for wandb prints to log.' 
            ' (default: 100)')
    args = parser.parse_args()
    ## debugging args - Keren
    if DEBUG is True:
        debug.set_args(args)

    # ddp setup
    # rank is the device
    if args.batch_size is None:
        args.batch_size = int(args.num_imgs / args.nproc)
    dtype = torch.FloatTensor
       
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
        args.folder = f"{args.dataset}{args.image_size}_{args.class_idx}_{str(args.num_imgs)}imgs_{str(args.sigma)}noise_std_dropout{args.dropout_val}_layer_size{args.layer_size}x{args.num_layer_decoder}_latent{args.latent_dim}_nsamples{args.num_samples}_epochs{args.num_epochs}_bs{args.batch_size}_lr{args.lr}"
    else:
        args.folder = f"{args.dataset}{args.image_size}_{str(args.num_imgs)}imgs_{str(args.sigma)}noise_std_dropout{args.dropout_val}_layer_size{args.layer_size}x{args.num_layer_decoder}_latent{args.latent_dim}_nsamples{args.num_samples}_epochs{args.num_epochs}_bs{args.batch_size}_lr{args.lr}"
    if args.latent_type == "gmm_eye" or args.latent_type == "gmm_custom" or args.latent_type == "gmm_low_eye" or args.latent_type == "gmm_low":
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
    if args.rand_shift==True:
        args.folder += "_rand_shift"
    if args.total_variation: 
        args.folder += "_tv{args.total_variation}"
    if args.nproc > 1:
        args.folder += f"_ngpu{args.nproc}"
    if args.sigma_loss is not None:
        args.folder += f"_regFSig{args.sigma_loss}"
    if args.suffix != '':
        args.folder += f"_{args.suffix}"

    if not os.path.exists(f'./{args.sup_folder}/{args.folder}'):
        os.makedirs(f'./{args.sup_folder}/{args.folder}')
    if not os.path.exists(f'./{args.sup_folder}/{args.folder}/model_checkpoints'):
        os.makedirs(f'./{args.sup_folder}/{args.folder}/model_checkpoints')
    
    with open("{}/args.json".format(f'./{args.sup_folder}/{args.folder}'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # regularizers
    if args.sigma_loss is None:
        args.sigma_loss = args.sigma
    else:
        args.sigma_loss = args.sigma_loss * args.sigma
        
    if args.sigma is None and args.dataset in ("m87", "sagA_video"):
        if args.dataset=="sagA_video":
            print("load matrix of sigmas")
            sigma = np.load("./utils/sigma_ngEHT_sagA_video_.5x.npz", allow_pickle=True)
        elif args.dataset == "m87":
            print("load matrix of sigmas")
            sigma = np.load("./utils/sigma_EHT2025_m87_frame.npz", allow_pickle=True)
        
        args.sigma = []
        for i in range(args.num_imgs):
            args.sigma.append(torch.tensor(sigma['arr_0'][i][np.newaxis, :, np.newaxis]))#.to(device))

    mp.spawn(main_function, args=(args,), nprocs=args.nproc)