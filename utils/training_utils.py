import numpy as np
import torch
import torch.nn.functional as functional
import matplotlib.pyplot as plt
import os
import torch.distributed as dist
from torch.autograd import Variable
import torch.optim as optim
from torch import Tensor
from torch.utils.data import WeightedRandomSampler
from .data_utils import get_envelope
from .eht_utils import loss_angle_diff
from .vis_utils import latest_epoch_path
from torch.nn.parallel import DistributedDataParallel as DDP


def get_gmm_gen_params(models, generator, num_imgs, model_type, eps_fixed):
    params = []
    for kk in range(num_imgs):
        if model_type != "gmm_low" and model_type != "gmm_low_eye":
            mu, L = models[kk][0], models[kk][1]
            mu = Variable(mu, requires_grad=True)
            L = Variable(L, requires_grad=True)
            models[kk][0] = mu
            models[kk][1] = L
            params += [models[kk][0]]
            params += [models[kk][1]]
        elif model_type == "gmm_low" or model_type == "gmm_low_eye":
            if eps_fixed == True:
                mu, L, eps = models[kk][0], models[kk][1], models[kk][2]
                mu = Variable(mu, requires_grad=True)
                L = Variable(L, requires_grad=True)
                models[kk][0] = mu
                models[kk][1] = L
                models[kk][2] = eps
                params += [models[kk][0]]
                params += [models[kk][1]]
            else:
                mu, L, eps = models[kk][0], models[kk][1], models[kk][2]
                mu = Variable(mu, requires_grad=True)
                L = Variable(L, requires_grad=True)
                eps = Variable(eps, requires_grad=True)
                models[kk][0] = mu
                models[kk][1] = L
                models[kk][2] = eps
                params += [models[kk][0]]
                params += [models[kk][1]]
                params += [models[kk][2]]
    params += generator.parameters()
    return params


class GMM_Custom(torch.distributions.MultivariateNormal):
    def __init__(self, mu, L, eps, device, latent_dim, **kwargs):
        C = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(eps)
        self.eps = eps
        self.L = L
        torch.distributions.MultivariateNormal.__init__(self, mu, C, **kwargs)
        
#         self.latent_dim = C.shape[0]
#         self.mu = mu
#         self.C = C
      
    def sample(self, n):
        latent_dim = self.covariance_matrix.shape[0]
#         x = torch.rand(n[0], latent_dim).to(self.loc.device)
#         x = torch.matmul(x, self.L.t())
#         x = x + self.loc.unsqueeze(0)

        x = torch.rand(latent_dim, n[0]).to(self.loc.device)
        x = torch.matmul(self.L, x)
        x = x.t() + self.loc.unsqueeze(0)
        return x


class Trainer():
    def __init__(
            self,
            generator,
            train_data,
            gpu_id,
            snapshot_path,
            sigma,
            targets,
            true_imgs,
            As,
            models,
            generator_func,
            args # org params before class
    ):
        self.gpu_id = gpu_id
        self.device = gpu_id # reduandant, remove later!
        self.generator = generator.to(gpu_id)
        if isinstance(gpu_id, int):
            self.generator = DDP(generator, device_ids=[gpu_id])
        self.train_data = train_data
        self.save_every = args.save_every
        self.num_imgs = args.num_imgs
        self.suffix = args.suffix
        self.world_size = args.nproc
        self.snapshot_path = snapshot_path
        # org params before class
        self.models = models
        self.generator_func = generator_func
        self.generator_type = args.generator_type
        self.lr = args.lr
        self.sigma = sigma
        self.targets = targets
        self.true_imgs = true_imgs
        self.num_samples = args.num_samples
        self.num_imgs_show = args.num_imgs_show
        self.num_epochs = args.num_epochs
        self.As = As
        self.task = args.task
        self.save_img = args.save_img
        self.dropout_val = args.dropout_val
        self.layer_size = args.layer_size
        self.num_layer_decoder = args.num_layer_decoder
        self.batchGD = args.batchGD
        self.dataset = args.dataset
        self.class_idx = args.class_idx
        self.seed = args.seed
        self.GMM_EPS = args.GMM_EPS
        self.sup_folder = args.sup_folder
        self.folder = args.folder
        self.latent_model = args.latent_type
        self.image_size = args.image_size
        self.num_channels = args.num_channels
        self.no_entropy = args.no_entropy
        self.eps_fixed = args.eps_fixed
        self.latent_dim = args.latent_dim
        self.epochs_run = 0
        self.gamma = args.gamma
        self.cphase_anneal = args.cphase_anneal         # None
        self.cphase_scale = args.cphase_scale           # None
        self.envelope_params = args.envelope            # 1
        self.centroid_params = args.centroid            # None
        self.locshift_params = args.locshift            # None
        self.from_checkpoint = args.from_checkpoint     # False
        self.validate_args = args.validate_args         # False
        self.normalize_loss = args.normalize_loss       # False
        torch.set_default_dtype(torch.float32)
        self.params = get_gmm_gen_params(self.models, self.generator, \
                                         self.num_imgs, self.latent_model, self.eps_fixed)
        self.optimizer = optim.Adam(self.params, lr=self.lr)

    def get_latent_model(self):
        list_of_models = [[torch.randn((self.latent_dim,)).to(self.device),
                           torch.tril(torch.ones((self.latent_dim, self.latent_dim))).to(self.device)] for i in range(self.num_imgs)]
        return list_of_models

    def get_avg_std_img(self, model, nimg = 40):
        
        #     if self.latent_model == 'flow':

        if (self.latent_model == 'gmm') or (self.latent_model == "gmm_eye"):
            prior = torch.distributions.MultivariateNormal(model[0],
                                                           (self.GMM_EPS) * torch.eye(self.latent_dim).to(self.device) + model[1] @ (
                                                               model[1].t()))
            z_sample = prior.sample((nimg,)).to(self.device)
        elif (self.latent_model == 'gmm_low') or (self.latent_model == "gmm_low_eye"):
            prior = torch.distributions.LowRankMultivariateNormal(model[0], model[1], model[2] * model[2] + 1e-6)
            z_sample = prior.sample((nimg,)).to(self.device)
        elif (self.latent_model == "gmm_custom"):
            prior = GMM_Custom(model[0], model[1], self.GMM_EPS, self.device, self.latent_dim)
            # (self.GMM_EPS)*torch.eye(self.latent_dim).to(self.device)+model[1]@(model[1].t()))
            z_sample = prior.sample((nimg,)).to(self.device)

        if self.generator_type == "norm_flow":
            img, _ = self.generator(z_sample)
            img = img.reshape([nimg, self.image_size, self.image_size])
        else:
            img = self.generator(z_sample)
        avg = torch.mean(img, dim=0)
        std = torch.std(img, dim=0)

        return avg, std
    
    
    def forward_model(self, A, x, task, idx=0, dataset=None, use_envelope=False):
        if task == 'phase-retrieval':
            y_complex = torch.fft.fft2(A[1] * x)
            # xhat = torch.zeros((x.shape[0], x.shape[2]*x.shape[2], 2)).to(device)
            # xhat[:,:,0] = x.reshape(x.shape[0], x.shape[2]*x.shape[2])
            # x_comp_vec = torch.view_as_complex(xhat)
            # y_complex = torch.mm(A, x_comp_vec.T)
            y_mag = y_complex.abs()
            y = y_mag
        elif task == 'gauss-phase-retrieval':
            xhat = torch.zeros((x.shape[0], x.shape[2] * x.shape[2], 2)).to(self.device)
            xhat[:, :, 0] = x.reshape(x.shape[0], x.shape[2] * x.shape[2])
            x_comp_vec = torch.view_as_complex(xhat)
            y_complex = torch.mm(A, x_comp_vec.T)
            y_mag = y_complex.abs()
            y = y_mag
        elif task == 'phase-problem':
            y_complex = torch.fft.fft2(A[1] * x)
            y = y_complex.angle()
        elif task == 'compressed-sensing' and dataset == "sagA_video":
            y = torch.einsum('ab,bcd->acd', x.reshape(x.shape[0], x.shape[2] * x.shape[2]), A[idx])
        elif task == 'compressed-sensing' and dataset != "sagA_video":
            y = torch.einsum('ab,bcd->acd', x.reshape(x.shape[0], x.shape[2] * x.shape[2]), A)
        elif task == 'super-res':
            y = A(x)
        elif task == 'inpainting':
            y = A * x
        elif task == 'closure-phase':
            if dataset == 'm87' or dataset == 'sagA':
                y = A(x, use_envelope=use_envelope)
            elif dataset == 'sagA_video':
                y = A(x, idx=idx, use_envelope=use_envelope)
            else:
                raise ValueError('invalid dataset for task closure-phase')
        return y

    def loss_data_fit(self, x, y, sigma, A, task, dataset, idx, gamma=None, cp_scale=1, use_envelope=False):
        mse = torch.nn.MSELoss()

        if task == 'denoising':
            loss = 0.5 * torch.sum((x - y) ** 2 / sigma ** 2, (-1, -2))
            #print(f'loss data shape {loss.shape}')
        elif task == 'phase-retrieval':
            meas = self.forward_model(A, x, task)
            loss = 0.5 * torch.sum((meas - y) ** 2 / (sigma * A[0]) ** 2, (-1, -2))
        elif task == 'gauss-phase-retrieval':
            meas = self.forward_model(A, x, task)
            loss = 0.5 * torch.sum((meas - y) ** 2 / (sigma) ** 2, (-1, -2))
        elif task == 'closure-phase':
            if gamma is None:
                raise ValueError('missing gamma for task closure-phase')

            meas_mag, meas_phase = self.forward_model(A, x, task, idx=idx, dataset=dataset,
                                                 use_envelope=use_envelope)
            y_mag, y_phase = y

            sigma_v, sigma_cp = sigma
            if hasattr(sigma_v, "__len__"):
                loss_mag = 0.5 * torch.sum((meas_mag - y_mag) ** 2 / sigma_v[idx] ** 2, -1)
            else:
                loss_mag = 0.5 * torch.sum((meas_mag - y_mag) ** 2 / sigma_v ** 2, -1)
            if hasattr(sigma_cp, "__len__"):
                loss_phase = loss_angle_diff(y_phase, meas_phase, sigma_cp[idx])
            else:
                loss_phase = loss_angle_diff(y_phase, meas_phase, sigma_cp)

            loss_mag_scaled = gamma * loss_mag
            loss_phase_scaled = cp_scale * meas_mag.shape[-1] / meas_phase.shape[-1] * loss_phase
            loss = loss_mag_scaled + loss_phase_scaled

            return loss, loss_mag_scaled.detach(), loss_phase_scaled.detach()
        else:
            meas = self.forward_model(A, x, task, idx=idx, dataset=dataset)
            if (dataset == "sagA_video" or dataset == "m87") and hasattr(sigma, "__len__"):
                #             print("new loss")
                loss = 0.5 * torch.sum((meas - y) ** 2 / sigma[idx] ** 2, (-1, -2))
            else:
                loss = 0.5 * torch.sum((meas - y) ** 2 / sigma ** 2, (-1, -2))
        return loss

    def loss_center(self, device, center=15.5, dim=32):
        # image prior - centering loss
        X = np.concatenate([np.arange(dim).reshape((1, dim))] * dim, 0)
        Y = np.concatenate([np.arange(dim).reshape((dim, 1))] * dim, 1)
        X = torch.Tensor(X).type(torch.float32).to(device=self.device)
        Y = torch.Tensor(Y).type(torch.float32).to(device=self.device)

        def func(y_pred):
            y_pred_flux = torch.mean(y_pred, (-1, -2))
            xc_pred_norm = torch.mean(y_pred * X, (-1, -2)) / y_pred_flux
            yc_pred_norm = torch.mean(y_pred * Y, (-1, -2)) / y_pred_flux

            loss = 0.5 * ((xc_pred_norm - center) ** 2 + (yc_pred_norm - center) ** 2)
            return loss[:, 0]

        return func

    # NOTE: the values are hardcoded rn for 12 filters (assuming batch size 12)
    def get_loc_shift_mats(self, s, d, etype='sq', r=3, all_locs=False):
        if not all_locs:
            p = s // 2 - d
            X = [p, p + d // 3, p + 2 * d // 3, s - p]
            Y = [p, s // 2, s - p]
            centers = [(x, y) for x in X for y in Y]
        else:
            centers = [(x, y) for x in range(s) for y in range(s)]
        num_filters = len(centers)
        filters = np.zeros((num_filters, 1, s, s))
        for i in range(num_filters):
            x, y = centers[i]
            filters[i, :, x, y] = 1
        filters = Tensor(filters).to(self.device)

        envelope = get_envelope(image_size=s, etype=etype, ds1=d, ds2=d + r, device=self.gpu_id)
        return filters, envelope

    def _save_checkpoint(self, epoch):
        ckp = self.generator.module.state_dict()
        PATH = self.snapshot_path
        torch.save(ckp, PATH)
        #print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    def init_training(self,
                        loss_list,
                        loss_data_list,
                        loss_prior_list,
                        loss_ent_list,
                        loss_mag_list,
                        loss_phase_list,
                        loss_centroid_list):
        # Initialize training params (+ load from checkpoint if specified)
        if self.from_checkpoint:
            pattern = 'cotrain-generator_*.pt'
            checkpoints_path = f'./{self.sup_folder}/{self.folder}/model_checkpoints/'
            last_generator_path = latest_epoch_path(checkpoints_path, pattern=pattern)

            pattern_begin, pattern_end = pattern.split('*')
            self.epochs_run = int(os.path.split(last_generator_path)[-1][len(pattern_begin):-len(pattern_end)])

            last_generator_sd = torch.load(last_generator_path)
            last_optimizer_sd = torch.load(last_generator_path.replace('cotrain-generator', 'optimizer'))
            self.generator.load_state_dict(last_generator_sd)

            self.models = []
            for i in range(self.num_imgs):
                latent_model_path = os.path.join(checkpoints_path, f'gmm-{i}.npz')
                latent_model_ = np.load(latent_model_path)
                self.models.append([Tensor(latent_model_[f]).to(self.device) for f in latent_model_.files])

            self.params = get_gmm_gen_params(self.models, self.generator, self.num_imgs, self.latent_model, self.eps_fixed)
            self.optimizer = optim.Adam(self.params, lr=self.lr)
            self.optimizer.load_state_dict(last_optimizer_sd)

            for (loss_checkpt_file, loss_checkpt_list) in [
                ('loss.npy', loss_list),
                ('loss_data.npy', loss_data_list),
                ('loss_prior.npy', loss_prior_list),
                ('loss_ent.npy', loss_ent_list),
                ('loss_mag.npy', loss_mag_list),
                ('loss_phase.npy', loss_phase_list),
                ('loss_centroid.npy', loss_centroid_list)
            ]:
                loss_checkpt_path = f'{self.sup_folder}/{self.folder}/{loss_checkpt_file}'
                if os.path.exists(loss_checkpt_path):
                    loss_checkpt_list[:] = np.load(loss_checkpt_path).tolist()

        # Set up optional params for closure phase problem

        # cphase anneal
        phase_anneal = np.ones(self.num_epochs)
        if self.cphase_anneal is not None and self.cphase_anneal > 0:
            n_anneal = 10000
            phase_anneal[self.cphase_anneal: self.cphase_anneal + n_anneal] = np.linspace(1, 0, num=n_anneal)
            phase_anneal[self.cphase_anneal + n_anneal:] = 0

        # envelope
        use_envelope = True if self.envelope_params is not None else False

        # centroid
        loss_centroid_fit = self.loss_center(device=self.device, center=self.image_size / 2 - 0.5, dim=self.image_size)

        if self.centroid_params:
            centroid_loss_wt, anneal_epoch = self.centroid_params
            centroid_anneal = np.ones(self.num_epochs)
            if anneal_epoch is not None:
                n_anneal = 10000
                e1, e2 = anneal_epoch, anneal_epoch + n_anneal
                centroid_anneal[e1:e2] = np.linspace(1, 0, num=e2 - e1)
                centroid_anneal[e2:] = 0
        else:
            centroid_anneal = None
            centroid_loss_wt = None

        # location shift w/ convolutions
        if self.locshift_params:
            etype, ds1, ds2, learn_locshift = self.locshift_params
            loc_shift = self.get_loc_shift_mats(self.image_size, d=ds1, etype=etype, r=ds2 - ds1,
                                           all_locs=learn_locshift)
        else:
            loc_shift = None
            learn_locshift = False

        if learn_locshift:
            prob_locations = (1 / self.image_size) ** 2 * torch.ones(self.image_size * self.image_size)
            prob_locations = Variable(prob_locations, requires_grad=True)
            self.params.append(prob_locations) # seems like a bug, since unused
        else:
            prob_locations = None
        return loss_centroid_fit, phase_anneal, centroid_anneal, centroid_loss_wt, loc_shift, learn_locshift, prob_locations, use_envelope

    def run_epoch_multi(self,
                    loss_sum,
                    loss_data_sum,
                    loss_prior_sum,
                    loss_ent_sum):
        for i in range(self.num_imgs // 2):
            if self.batchGD == False:
                self.optimizer.zero_grad()
            target = self.targets[i]

            if (self.latent_model == 'gmm') or (self.latent_model == "gmm_eye"):
                mu, L = self.models[i]
                spread_cov = (L @ (L.t())).to(self.device) + torch.diag(torch.ones(self.latent_dim)).to(self.device) * (self.GMM_EPS)
                prior = torch.distributions.MultivariateNormal(mu, spread_cov)
            elif (self.latent_model == 'gmm_low') or (self.latent_model == "gmm_low_eye"):
                mu, L, eps = self.models[i]
                prior = torch.distributions.LowRankMultivariateNormal(mu, L, eps * eps + 1e-6)
            elif (self.latent_model == "gmm_custom"):
                mu, L = self.models[i]
                #                     spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
                prior = GMM_Custom(mu, L, self.GMM_EPS, self.device, self.latent_dim)

            z_sample = prior.sample((self.num_samples,)).to(self.device)

            if self.generator_type == "norm_flow":
                img, logdet = self.generator_func(z_sample)
                img = img.reshape([self.num_samples, self.image_size, self.image_size])
            else:
                img = self.generator_func(z_sample)

            if self.no_entropy == True:
                log_ent = 0
            else:
                if self.generator_type == "norm_flow":
                    log_ent = prior.log_prob(z_sample)  # + logdet
                else:
                    log_ent = prior.log_prob(z_sample)

            loss_prior = 0.5 * torch.sum(z_sample ** 2, 1)
            if 'multi-denoising' in self.task:
                curr_task = 'denoising'
            if 'multi-compressed-sensing' in self.task:
                curr_task = 'compressed-sensing'
            loss_data = self.loss_data_fit(img, target, self.sigma[0], self.As[0], curr_task, idx=i, dataset=self.dataset,
                                      gamma=self.gamma)

            loss_denoise = torch.mean(loss_data + log_ent + loss_prior)

            loss_sum = loss_sum + loss_denoise
            loss_data_sum = loss_data_sum + torch.mean(loss_data)
            loss_prior_sum = loss_prior_sum + torch.mean(loss_prior)
            if self.no_entropy == False:
                loss_ent_sum = loss_ent_sum + torch.mean(log_ent)

            if self.batchGD == False:
                loss_denoise.backward(retain_graph=True)
                self.optimizer.step()
        for j in range(self.num_imgs // 2 + 1, self.num_imgs):
            if self.batchGD == False:
                self.optimizer.zero_grad()
            target = self.targets[j]

            if (self.latent_model == 'gmm') or (self.latent_model == "gmm_eye"):
                mu, L = self.models[j]
                spread_cov = (L @ (L.t())).to(self.device) + torch.diag(torch.ones(self.latent_dim)).to(self.device) * (self.GMM_EPS)
                prior = torch.distributions.MultivariateNormal(mu, spread_cov)
            elif (self.latent_model == 'gmm_low') or (self.latent_model == "gmm_low_eye"):
                mu, L, eps = self.models[j]
                prior = torch.distributions.LowRankMultivariateNormal(mu, L, eps * eps + 1e-6)
            elif (self.latent_model == "gmm_custom"):
                mu, L = self.models[j]
                spread_cov = (L @ (L.t())).to(self.device) + torch.diag(torch.ones(self.latent_dim)).to(self.device) * (self.GMM_EPS)
                prior = GMM_Custom(mu, L, self.GMM_EPS, self.device, self.latent_dim)
            z_sample = prior.sample((self.num_samples,)).to(self.device)

            if self.generator_type == "norm_flow":
                img, logdet = self.generator_func(z_sample)
                img = img.reshape([self.num_samples, self.image_size, self.image_size])
            else:
                img = self.generator_func(z_sample)

            if self.no_entropy == True:
                log_ent = 0
            else:
                if self.generator_type == "norm_flow":
                    log_ent = prior.log_prob(z_sample)  # + logdet
                else:
                    log_ent = prior.log_prob(z_sample)

            loss_prior = 0.5 * torch.sum(z_sample ** 2, 1)
            if 'denoising-compressed-sensing' in self.task:
                curr_task = 'compressed-sensing'
            elif 'compressed-sensing-phase-retrieval' or 'denoising-phase-retrieval' in self.task:
                if 'gauss' in self.task:
                    curr_task = 'gauss-phase-retrieval'
                else:
                    curr_task = 'phase-retrieval'
            elif 'phase-problem' in self.task:
                curr_task = 'phase-problem'
            loss_data = self.loss_data_fit(img, target, self.sigma[1], self.As[1], curr_task, idx=j, dataset=self.dataset,
                                      gamma=self.gamma)

            loss_new_task = torch.mean(loss_data + log_ent + loss_prior)

            loss_sum = loss_sum + loss_new_task
            loss_data_sum = loss_data_sum + torch.mean(loss_data)
            loss_prior_sum = loss_prior_sum + torch.mean(loss_prior)
            if self.no_entropy == False:
                loss_ent_sum = loss_ent_sum + torch.mean(log_ent)

            if self.batchGD == False:
                loss_new_task.backward(retain_graph=True)
                self.optimizer.step()
        if self.batchGD == True:
            loss_sum.backward(retain_graph=True)
            self.optimizer.step()

        return loss_sum, loss_data_sum, loss_prior_sum, loss_ent_sum

    def run_epoch_other(self,
                    k,
                    loss_sum, loss_data_sum, loss_prior_sum, loss_ent_sum, loss_mag_sum, loss_phase_sum, loss_centroid_sum,
                    loss_centroid_fit, phase_anneal, centroid_anneal, centroid_loss_wt, loc_shift, learn_locshift, prob_locations, use_envelope):
        b_sz = len(next(iter(self.train_data))[0])
        if k % 50 == 0:
            print(f"[GPU{self.gpu_id}] Epoch {k} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        i = 0
        for models, idx, targets in self.train_data:
            print(f'gpu{self.gpu_id}, idx={idx}')
            for j in idx:
                ind = j.item()
                model = self.models[ind]#.to(self.gpu_id)
                target = self.targets[ind].to(self.gpu_id)
                # if self.gpu_id == 0:
                #     filename = f'./figures/target_train_{i}.png'
                #     plt.imshow(targets.cpu().numpy().squeeze())
                #     plt.figure().savefig(filename)
                #     i+=1
                # print('inside train_latent_gmm_and_generator')
                # print('targets')
                # print(f'{targets}')
                # print('models')
                # print(f'{models}')
                if self.batchGD == False:
                    self.optimizer.zero_grad()
                #target = targets[0]
                #model = models[0]

            #                 mu, L = models[i]
            #                 spread_cov = (L@(L.t())).to(device) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
            #                 prior = torch.distributions.MultivariateNormal(mu, spread_cov)

                if (self.latent_model == 'gmm') or (self.latent_model == "gmm_eye"):
                    # if b_sz == 1:
                    #     mu, L = torch.split(models, [1, self.latent_dim], dim=1)
                    #     mu = mu.squeeze(0).squeeze(0)
                    #     L = L.squeeze(0)
                    #     spread_cov = (L @ (L.t())).to(self.device) + torch.diag(torch.ones(self.latent_dim)).to(self.device) * (self.GMM_EPS)
                    #     prior = torch.distributions.MultivariateNormal(mu, spread_cov, validate_args=self.validate_args)
                    # else:
                    #     prior = []
                    #     for ind in idx:
                    #     # mu, L = torch.split(models, [1, self.latent_dim], dim=1)
                    #     # mu = mu.squeeze(1)
                    #     # L_T = torch.transpose(L, 1, 2)
                    #     # diagonal_ones = torch.diag(torch.ones(self.latent_dim))
                    #     # expanded_diagonal_ones = diagonal_ones.unsqueeze(0).expand(b_sz, self.latent_dim, self.latent_dim)
                    #     # spread_cov = (L @ (L_T)).to(self.device) + expanded_diagonal_ones.to(self.device) * (self.GMM_EPS)
                    #     # prior = torch.distributions.MultivariateNormal(mu, spread_cov, validate_args=self.validate_args)
                    #mu, L = torch.split(models[ind], [1, self.latent_dim], dim=0)
                    mu, L = model[0].to(self.device), model[1].to(self.device)
                    #mu = mu.squeeze(0)
                    spread_cov = (L @ (L.t())).to(self.device) + torch.diag(torch.ones(self.latent_dim)).to(self.device) * (self.GMM_EPS)
                    prior = torch.distributions.MultivariateNormal(mu, spread_cov, validate_args=self.validate_args)

                elif (self.latent_model == 'gmm_low') or (self.latent_model == "gmm_low_eye"):
                    mu, L, eps = models[i]
                    #                     print(eps*eps)
                    prior = torch.distributions.LowRankMultivariateNormal(mu, L, eps * eps + 1e-6,
                                                                          validate_args=self.validate_args)
                elif (self.latent_model == "gmm_custom"):
                    mu, L = models[i]
                    spread_cov = (L @ (L.t())).to(self.device) + torch.diag(torch.ones(self.latent_dim)).to(self.device) * (self.GMM_EPS)
                    prior = GMM_Custom(mu, L, self.GMM_EPS, self.device, self.latent_dim, validate_args=self.validate_args)

                z_sample = prior.sample((self.num_samples,)).to(self.device)
    
                if self.generator_type == "norm_flow":
                    img, logdet = self.generator_func(z_sample)
                    img = img.reshape([self.num_samples, self.image_size, self.image_size])
                else:
                    # if b_sz == 1:
                    img = self.generator_func(z_sample)
                    # else:
                    #     img = torch.zeros(b_sz, 12, 1, self.image_size, self.image_size)
                    #     for j in range(b_sz):
                    #         img[j] = self.generator_func(z_sample[:, j, :].squeeze(1))
                    #[:, 0, :]
                    # img = self.generator_func(z_sample)
                # print(img.shape, z_sample.shape, img.shape)
    
                if self.no_entropy == True:
                    log_ent = 0
                else:
                    if self.generator_type == "norm_flow":
                        log_ent = prior.log_prob(z_sample)  # + logdet
                    else:
                        log_ent = prior.log_prob(z_sample)
    
                loss_prior = 0.5 * torch.sum(z_sample ** 2, 1)
    
                if loc_shift is not None:
                    filters, envelope = loc_shift
                    if learn_locshift:
                        samples = WeightedRandomSampler(prob_locations, img.shape[0], replacement=True)
                        samples = list(samples)
                        filters = filters[samples]
    
                    img = envelope * img
                    img = functional.conv2d(filters.transpose(0, 1), img.flip((3)).flip((2)),
                                            padding='same', groups=filters.shape[0]).transpose(0, 1)
                #img = 
                loss_data = self.loss_data_fit(img, target, self.sigma, self.As, self.task, idx=i, dataset=self.dataset,
                                          gamma=self.gamma, cp_scale=self.cphase_scale * phase_anneal[k],
                                          use_envelope=use_envelope)

                # if normalize_loss:
                # loss_prior = torch.divide(loss_prior, latent_dim)
                # loss_data = torch.divide(loss_data, image_size * image_size)
    
                if 'closure-phase' in self.task:
                    loss_data, loss_mag, loss_phase = loss_data
    
                if self.centroid_params:
                    loss_centroid = centroid_anneal[k] * centroid_loss_wt * loss_centroid_fit(img)
                    loss = torch.mean(loss_data + log_ent + loss_prior + loss_centroid)
                else:
                    loss = torch.mean(loss_data + log_ent + loss_prior)

                loss_sum = loss_sum + loss
                loss_data_sum = loss_data_sum + torch.mean(loss_data)
                loss_prior_sum = loss_prior_sum + torch.mean(loss_prior)
    
                if 'closure-phase' in self.task:
                    loss_mag_sum += torch.mean(loss_mag)
                    loss_phase_sum += torch.mean(loss_phase)
                if self.centroid_params:
                    loss_centroid_sum += torch.mean(loss_centroid)
                if self.no_entropy == False:
                    loss_ent_sum = loss_ent_sum + torch.mean(log_ent)
    
                if self.batchGD == False:
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
            i += 1
        if self.batchGD == True:
            loss_sum.backward(retain_graph=True)
            self.optimizer.step()

        # cur_loss = loss.item()
        return loss_sum, loss_data_sum, loss_prior_sum, \
                loss_ent_sum, loss_mag_sum, loss_phase_sum, \
                loss_centroid_sum

    def run_epoch(self,
                  k,
                  loss_centroid_fit,
                  phase_anneal,
                  centroid_anneal,
                  centroid_loss_wt,
                  loc_shift,
                  learn_locshift,
                  prob_locations,
                  use_envelope):
        loss_sum = 0
        loss_data_sum = 0
        loss_prior_sum = 0
        loss_ent_sum = 0
        loss_mag_sum = 0
        loss_phase_sum = 0
        loss_centroid_sum = 0

        b_sz = len(next(iter(self.train_data))[0])
        if isinstance(self.device, int):
            self.train_data.sampler.set_epoch(k)
        self.optimizer.zero_grad()
        if 'multi' in self.task:
            # Maybe wrong inside, don't really care
            loss_sum, loss_data_sum, loss_prior_sum, loss_ent_sum \
                = self.run_epoch_multi(loss_sum,
                                   loss_data_sum,
                                   loss_prior_sum,
                                   loss_ent_sum)
        else:
            loss_sum, loss_data_sum, loss_prior_sum,\
            loss_ent_sum, loss_mag_sum, loss_phase_sum,\
            loss_centroid_sum = self.run_epoch_other(k,
                                                 loss_sum,
                                                 loss_data_sum,
                                                 loss_prior_sum,
                                                 loss_ent_sum,
                                                 loss_mag_sum,
                                                 loss_phase_sum,
                                                 loss_centroid_sum,
                                                 loss_centroid_fit, phase_anneal, centroid_anneal, centroid_loss_wt, loc_shift, learn_locshift, prob_locations, use_envelope)
        return loss_sum, loss_data_sum, loss_prior_sum, \
            loss_ent_sum, loss_mag_sum, loss_phase_sum, \
            loss_centroid_sum

    def train_latent_gmm_and_generator(self):

        loss_list = []
        loss_data_list = []
        loss_mean_true_list = []
        loss_prior_list = []
        loss_ent_list = []
        loss_mag_list = []
        loss_phase_list = []
        loss_centroid_list = []

        loss_centroid_fit, phase_anneal, centroid_anneal, centroid_loss_wt, loc_shift, learn_locshift, prob_locations, use_envelope\
                = self.init_training(loss_list,
                        loss_data_list,
                        loss_prior_list,
                        loss_ent_list,
                        loss_mag_list,
                        loss_phase_list,
                        loss_centroid_list)

        # Training loop
        for k in range(self.epochs_run, self.num_epochs):
            loss_sum, loss_data_sum, loss_prior_sum,\
            loss_ent_sum, loss_mag_sum, loss_phase_sum,\
            loss_centroid_sum = self.run_epoch(k,
                                               loss_centroid_fit,
                                               phase_anneal,
                                               centroid_anneal,
                                               centroid_loss_wt,
                                               loc_shift,
                                               learn_locshift,
                                               prob_locations,
                                               use_envelope)
            if isinstance(self.gpu_id, int):
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_data_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_prior_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_ent_sum, op=dist.ReduceOp.SUM)
                if 'closure-phase' in self.task:
                    dist.all_reduce(loss_mag_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss_phase_sum, op=dist.ReduceOp.SUM)
                if self.centroid_params:
                    dist.all_reduce(loss_centroid_sum, op=dist.ReduceOp.SUM)

            if self.gpu_id == 0 or (isinstance(self.gpu_id, int) != True):
                if self.normalize_loss:
                    # might be different normalization
                    loss_list.append(loss_sum.item() / self.num_imgs)
                    loss_data_list.append(loss_data_sum.item() / self.num_imgs)
                    loss_prior_list.append(loss_prior_sum.item() / self.num_imgs)
                else:
                    loss_list.append(loss_sum.item())
                    loss_data_list.append(loss_data_sum.item())
                    loss_prior_list.append(loss_prior_sum.item())
                if 'closure-phase' in self.task:
                    loss_mag_list.append(loss_mag_sum.item())
                    loss_phase_list.append(loss_phase_sum.item())
                if self.centroid_params:
                    loss_centroid_list.append(loss_centroid_sum.item())
                if self.no_entropy == False:
                    if self.normalize_loss:
                        loss_ent_list.append(loss_ent_sum.item() / self.num_imgs)
                    else:
                        loss_ent_list.append(loss_ent_sum.item())
                else:
                    loss_ent_list.append(loss_ent_sum)
                
                # mean_true_diff_loss = self.get_mu_loss()
                # loss_mean_true_list.append(mean_true_diff_loss)
                
                # self.plot_epoch_results(k, loss_sum, loss_data_sum, loss_prior_sum,
                #         loss_ent_sum, loss_mag_sum, loss_phase_sum, loss_centroid_sum,
                #         self.targets, self.models, loc_shift, prob_locations)
    
            if self.gpu_id == 0 and k % self.save_every == 0:
                self._save_checkpoint(k)
                self.checkpoint_results(self.latent_dim, self.generator_func, self.models, str(k), self.num_imgs,
                                   self.GMM_EPS, self.folder, self.sup_folder, self.latent_model, self.image_size, self.generator_type)

                self.save_model_gen_params(self.generator, self.models, self.optimizer, str(k), self.num_imgs, self.folder,
                                      self.sup_folder, self.latent_model)
        # save data
        if self.gpu_id == 0 or self.gpu_id.type == 'cuda':
            self.get_statistics(loss_data_list, loss_prior_list, loss_list, loss_mag_list, loss_phase_list,
                    loss_centroid_list, loss_ent_list, loss_mean_true_list, self.targets, self.models)
        #return self.models, self.generator

    def plot_epoch_results(self, k, loss_sum, loss_data_sum, loss_prior_sum,
                        loss_ent_sum, loss_mag_sum, loss_phase_sum, loss_centroid_sum,
                        org_targets, org_models, loc_shift, prob_locations):
        if k % 50 == 0:
            print("-----------------------------")
            print("Epoch {}".format(k))
            # print("Curr ELBO: {}".format(cur_loss))
            print("Loss all: {:e}".format(loss_sum.item() / self.num_imgs))
            print("Loss data fit: {:e}".format(loss_data_sum.item() / self.num_imgs))
            if 'closure-phase' in self.task:
                print(f"Scaled loss magnitude: {loss_mag_sum.item() / self.num_imgs:e} / "
                      f"phase: {loss_phase_sum.item() / self.num_imgs:e}")
            if self.centroid_params:
                print(f"Loss centroid: {loss_centroid_sum.item() / self.num_imgs:e}")
            print("Loss prior: {}".format(loss_prior_sum.item() / self.num_imgs))
            if self.no_entropy == False:
                print("Loss entropy: {}".format(loss_ent_sum.item() / self.num_imgs))
            else:
                print("Loss entropy: {}".format(loss_ent_sum / self.num_imgs))
            print("-----------------------------")
            if ((k < 500) and (k % 100 == 0)) or ((k >= 500) and (k % 1000 == 0)):
                if 'multi' in self.task:
                    img_indices = [i for i in range(self.num_imgs_show // 2)]
                    img_indices += [self.num_imgs // 2 + i for i in range(self.num_imgs_show)]
                else:
                    img_indices = range(self.num_imgs_show)
                # print("passed before avg_img_list")

                # avg_img_list = [self.get_avg_std_img(self.models[i])[0] \
                #     for i in img_indices]
                # std_img_list = [self.get_avg_std_img(self.models[i])[1] \
                #     for i in img_indices]
                avg_img_list = [torch.zeros(1, 1, 64).to(self.gpu_id) \
                    for i in img_indices]
                std_img_list = [torch.zeros(1, 1, 64).to(self.gpu_id) \
                    for i in img_indices]
                # print(f'avg_img_list len={len(avg_img_list)}')
                # print(f'std_img_list len={len(std_img_list)}')
                # print("passed all avg_img_list")
                
                self.plot_results(self.models, self.generator_func, self.true_imgs, org_targets, self.num_channels,
                              self.image_size, self.num_imgs_show, self.num_imgs,
                              self.sigma, avg_img_list, std_img_list, self.save_img, str(k),
                              self.dropout_val, self.layer_size, self.num_layer_decoder,
                              self.batchGD, self.dataset, self.folder, self.sup_folder, self.GMM_EPS,
                              self.task, self.latent_model, self.generator_type, envelope_params=self.envelope_params,
                              loc_shift=loc_shift, prob_locations=prob_locations)

    def get_statistics(self, loss_data_list, loss_prior_list, loss_list, loss_mag_list, loss_phase_list,
                        loss_centroid_list, loss_ent_list, loss_mean_true_list, org_targets, org_models):
           
        plt.figure()
        plt.plot(loss_mean_true_list, label="loss")
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('|mu-x|/num_pixels')
        plt.title('loss of mean vs true images')
        plt.savefig(f'./{self.sup_folder}/{self.folder}/loss mean true.png')
        plt.close()
    
        plt.figure()
        plt.plot(loss_list)
        plt.savefig(f'./{self.sup_folder}/{self.folder}/loss.png')
        plt.close()
    
        plt.figure()
        plt.plot(loss_list, label="all")
        plt.plot(loss_data_list, label="data")
        plt.plot(loss_prior_list, label="prior")
        plt.plot(loss_ent_list, label="ent")
        if 'closure-phase' in self.task:
            plt.plot(loss_mag_list, label='mag')
            plt.plot(loss_phase_list, label='phase')
        if self.centroid_params:
            plt.plot(loss_centroid_list, label='centroid')
        plt.legend()
        plt.savefig(f'./{self.sup_folder}/{self.folder}/loss_all.png')
        plt.yscale('log')
        plt.savefig(f'./{self.sup_folder}/{self.folder}/loss_all_log.png')
        plt.close()
    
        np.save(f'./{self.sup_folder}/{self.folder}/loss.npy', loss_list)
        np.save(f'./{self.sup_folder}/{self.folder}/loss_data.npy', loss_data_list)
        np.save(f'./{self.sup_folder}/{self.folder}/loss_prior.npy', loss_prior_list)
        np.save(f'./{self.sup_folder}/{self.folder}/loss_ent.npy', loss_ent_list)
        if 'closure-phase' in self.task:
            np.save(f'./{self.sup_folder}/{self.folder}/loss_mag.npy', loss_mag_list)
            np.save(f'./{self.sup_folder}/{self.folder}/loss_phase.npy', loss_phase_list)
        if self.centroid_params:
            np.save(f'./{self.sup_folder}/{self.folder}/loss_centroid.npy', loss_centroid_list)
    
        # Kerens
        b_sz = len(next(iter(self.train_data))[0])
        print(f'#loss points {len(loss_list)}')
        np.save(f'./{self.sup_folder}/{self.folder}/loss_ddp_{self.num_imgs}im_{b_sz}bs_{self.num_epochs}epochs_{self.world_size}gpus_{self.suffix}.npy', loss_list)
        plt.figure()
        plt.plot(loss_list)
        plt.savefig(f'./{self.sup_folder}/{self.folder}/loss_ddp_{self.num_imgs}im_{b_sz}bs_{self.num_epochs}epochs_{self.world_size}gpus_{self.suffix}.png')
        plt.close()


    def get_mu_loss(self):
        mean_true_diff = 0
        #I think this should be changed...
        for i in range(self.num_imgs):
            mu, L = torch.split(self.models[i], [1, self.latent_dim], dim=1)
            avg = self.generator_func(mu.unsqueeze(0)) 
            mean_true_diff_val = torch.abs(torch.sum(avg - self.true_imgs[i], (-1, -2))) / torch.norm(self.true_imgs[i])
            mean_true_diff_val = mean_true_diff_val.detach().cpu().numpy()[0][0]
            mean_true_diff += mean_true_diff_val
            #print(f'{i} mean_true_diff_val={mean_true_diff_val}')
        return mean_true_diff / self.num_imgs
    
        
        
        
      
        
        
    def plot_results(self, models, generator, true_imgs, noisy_imgs, num_channels,
                     image_size, num_imgs_show, num_imgs,
                     sigma, avg_img_list, std_img_list, save_img, epoch,
                     dropout_val, layer_size, num_layer_decoder,
                     batchGD, dataset, folder, sup_folder, GMM_EPS,
                     task, latent_model, generator_type, envelope_params=None,
                     loc_shift=None, prob_locations=None):

        num_samples = 7
    #     num_channels = true_imgs[0].shape[1]
        latent_dim = models[0][0].shape[0]
    #     image_size = true_imgs[0].shape[2]

        if envelope_params is not None:
            etype, ds1, ds2 = envelope_params
            envelope = get_envelope(image_size, etype=etype, ds1=ds1, ds2=ds2, device=self.gpu_id)
            envelope = envelope.detach().cpu().numpy().reshape((image_size, image_size))
        elif loc_shift is not None:
            _, envelope = loc_shift
            envelope = envelope.detach().cpu().numpy().reshape((image_size, image_size))
        else:
            envelope = np.ones((image_size, image_size))

        if 'multi' in task:
            img_indices = [i for i in range(num_imgs_show // 2)]
            img_indices += [num_imgs // 2 + i for i in range(num_imgs_show)]
        else:
            img_indices = range(num_imgs_show)
        fig,ax = plt.subplots(len(img_indices), num_samples + 7, figsize = (22,12))
        if not hasattr(sigma, '__len__'):
            fig.suptitle(task + ' with ' + str(num_imgs)+' images ' + str(sigma) + ' noise std')
        else:
            fig.suptitle(task + ' with ' + str(num_imgs) + ' images realistic noise std')
        kk = 0
        for ii in img_indices:
            true = true_imgs[ii].detach().cpu().numpy().reshape([num_channels, image_size, image_size])
            true = true[0,:,:].reshape([image_size, image_size])
            if 'denoising' in task and ii < num_imgs_show:
                noisy = self.targets[ii].detach().cpu().numpy().reshape([num_channels, image_size, image_size])
                noisy = noisy[0,:,:].reshape([image_size, image_size])
            else:
                noisy = np.zeros([image_size, image_size])
            ax[kk][0].imshow(true, cmap = "gray", vmin=0, vmax=1)
            ax[kk][0].axis("off")
            ax[kk][0].set_title("true (x)")
            ax[kk][1].imshow(noisy, cmap = "gray", vmin=0, vmax=1)
            ax[kk][1].axis("off")
            ax[kk][1].set_title("noisy (y)")


    #         mu, L = models[ii]
    #         spread_cov = L@(L.t()) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
    #         prior = torch.distributions.MultivariateNormal(mu, spread_cov)

            if (latent_model == 'gmm') or (latent_model == "gmm_eye"):
                mu, L = models[ii]
                spread_cov = (L@(L.t())).to(self.device) + torch.diag(torch.ones(latent_dim)).to(self.device)*(GMM_EPS)
                prior = torch.distributions.MultivariateNormal(mu, spread_cov)
            elif (latent_model == 'gmm_low') or (latent_model == "gmm_low_eye"):
                mu, L, eps = models[ii]
                prior = torch.distributions.LowRankMultivariateNormal(mu, L, eps*eps+1e-6)
            elif (latent_model == "gmm_custom"):
                mu, L = models[ii]
                spread_cov = (L@(L.t())).to(self.device) + torch.diag(torch.ones(latent_dim)).to(self.device)*(GMM_EPS)
                prior = GMM_Custom(mu, L, GMM_EPS, self.device, latent_dim)


            z_sample = prior.sample((num_samples,)).to(self.device)
            if generator_type == "norm_flow":
                samples_to_show,_ = generator(z_sample)
                samples_to_show = samples_to_show.reshape([num_samples, 1, image_size, image_size])
            else:
                samples_to_show = generator(z_sample)

            mean_i = avg_img_list[kk].detach().cpu().numpy().reshape([num_channels, image_size, image_size])
            mean_i = mean_i[0,:,:].reshape([image_size, image_size])
            mean_i = envelope * mean_i
            ax[kk][2].imshow(mean_i, cmap = "gray", vmin=0, vmax=1)
            ax[kk][2].axis("off")
            ax[kk][2].set_title("mean (mu)")
            std_i = std_img_list[kk].detach().cpu().numpy().reshape([num_channels, image_size, image_size])
            std_i = std_i[0,:,:].reshape([image_size, image_size])
            std_i = envelope * std_i
            ax[kk][3].imshow(std_i, cmap = "gray")
            ax[kk][3].axis("off")
            ax[kk][3].set_title("std")
            norm_err = (np.abs(mean_i-true)/np.linalg.norm(std_i))
            ax[kk][4].imshow(norm_err, cmap = "hot", vmin=0, vmax = 5)
            ax[kk][4].axis("off")
            ax[kk][4].set_title(f"|mu-x|/std \n {np.mean(norm_err):.3f}")
            if 'multi' in task:
                sig_val = sigma[0]
            elif (dataset in ("m87", "sagA_video", "sagA")) and hasattr(sigma, "__len__"):
                sig_val = 2#sigma[ii].detach().cpu().numpy()
            else:
                sig_val = sigma
            norm_err = (np.abs(mean_i-true)/sig_val)
            ax[kk][5].imshow(norm_err, cmap = "hot", vmin=0, vmax = 5)
            ax[kk][5].axis("off")
            ax[kk][5].set_title(f"|mu-x|/sigma \n {np.mean(norm_err):.3f} ")

            norm_err = (np.abs(mean_i-noisy)/sig_val)
            ax[kk][6].imshow(norm_err, cmap = "hot", vmin=0, vmax = 5)
            ax[kk][6].axis("off")
            ax[kk][6].set_title(f"|mu-y|/sigma \n {np.mean(norm_err):.3f}")

            for jj in range(num_samples):
                sample = samples_to_show[jj,:,:,:].detach().cpu().numpy().reshape([num_channels,image_size, image_size])
                sample = sample[0,:,:].reshape([image_size, image_size])
                sample = envelope * sample
                ax[kk][jj+7].imshow(sample, cmap='gray', vmin=0, vmax=1)
                ax[kk][jj+7].axis("off")
                std = np.around(np.sqrt(np.mean((mean_i - sample)**2)), 3)
                ax[kk][jj+7].set_title(str(std))
            kk += 1

        if save_img == True:
            folder_path = sup_folder + '/' + folder
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            plt.savefig(f'./{sup_folder}/{folder}/{str(epoch).zfill(7)}epochs.png')
            plt.close()

            if 'multi' in task:
                for i in range(len(self.targets)//2):
                    np.save(f"./{sup_folder}/{folder}/noisy_imgs_1.npy",
                            [self.targets[i].detach().cpu().numpy() for i in range(len(self.targets)//2)])
                for i in range(len(self.targets)//2,):
                    np.save(f"./{sup_folder}/{folder}/noisy_imgs_2.npy",
                            [self.targets[i].detach().cpu().numpy() for i in range(len(self.targets)//2,)])
            else:
                if dataset not in ("sagA_video", "m87", "sagA"):
                    np.save(f"./{sup_folder}/{folder}/noisy_imgs.npy",
                            [self.targets[i].detach().cpu().numpy() for i in range(len(self.targets))])
                elif task == 'closure-phase':
                    noisy_imgs_ = [(x[0].detach().cpu().numpy(), x[1].detach().cpu().numpy())
                                   for x in self.targets]
                    noisy_imgs_ = [item for sublist in noisy_imgs_ for item in sublist]
                    np.savez(f"./{sup_folder}/{folder}/noisy_imgs.npz",
                             *[noisy_imgs_[i] for i in range(len(noisy_imgs_))])
                else:
                    np.savez(f"./{sup_folder}/{folder}/noisy_imgs.npz",
                            *[self.targets[i].detach().cpu().numpy() for i in range(len(self.targets))])
            np.save(f"./{sup_folder}/{folder}/true_imgs.npy", [true_imgs[i].detach().cpu().numpy() for i in range(len(true_imgs))])

            if prob_locations is not None:
                prob_locs = prob_locations.detach().cpu().numpy().reshape((image_size, image_size))
                plt.imshow(prob_locs, cmap='gray')
                plt.savefig(f'./{sup_folder}/{folder}/{str(epoch).zfill(7)}prob_locs.png')
                plt.close()
                np.save(f'./{sup_folder}/{folder}/prob_locs.npy', prob_locs)

    def checkpoint_results(self, latent_dim, generator, models, epoch, num_imgs,
                           GMM_EPS, folder, sup_folder, latent_model, image_size, generator_type):

        if not os.path.exists(f"./{sup_folder}/{folder}/files"):
            os.makedirs(f"./{sup_folder}/{folder}/files")

        if not os.path.exists(f"./{sup_folder}/{folder}/files/{epoch.zfill(7)}"):
            os.makedirs(f"./{sup_folder}/{folder}/files/{epoch.zfill(7)}")

        ''' generate samples from generator'''
        s = 40
        temp = 1
        rand_samples = torch.randn([s, latent_dim]).to(self.device)*temp
        if generator_type == "norm_flow":
            samples,_ = generator(rand_samples)
            samples = samples.reshape([s, image_size, image_size])
        else:
            samples = generator(rand_samples)

        np.save(f"./{sup_folder}/{folder}/files/{epoch.zfill(7)}/generator_samples.npy", samples.detach().cpu().numpy())

        s = 40

        ''' plot each models samples'''
        for ii in range(num_imgs):
    #         mu, L = models[ii]
    #         spread_cov = L@(L.t()) + torch.diag(torch.ones(latent_dim)).to(device)*(GMM_EPS)
    #         prior = torch.distributions.MultivariateNormal(mu, spread_cov)#model[1]@(model[1].t()))

            if (latent_model == 'gmm') or (latent_model == "gmm_eye"):
                mu, L = models[ii]
                spread_cov = (L@(L.t())).to(self.device) + torch.diag(torch.ones(latent_dim)).to(self.device)*(GMM_EPS)
                prior = torch.distributions.MultivariateNormal(mu, spread_cov)
            elif (latent_model == 'gmm_low') or (latent_model == "gmm_low_eye"):
                mu, L, eps = models[ii]
                prior = torch.distributions.LowRankMultivariateNormal(mu, L, eps*eps+1e-6)
            elif (latent_model == "gmm_custom"):
                mu, L = models[ii]
                spread_cov = (L@(L.t())).to(self.device) + torch.diag(torch.ones(latent_dim)).to(self.device)*(GMM_EPS)
                prior = GMM_Custom(mu, L, GMM_EPS, self.device, latent_dim)

            z_sample = prior.sample((s,)).to(self.device)
            if generator_type == "norm_flow":
                samples,_ = generator(z_sample)
                samples = samples.reshape([s, image_size, image_size])
            else:
                samples = generator(z_sample)

            np.save(f"./{sup_folder}/{folder}/files/{epoch.zfill(7)}/xsamples_{str(ii).zfill(3)}.npy", samples.detach().cpu().numpy())
            np.save(f"./{sup_folder}/{folder}/files/{epoch.zfill(7)}/zsamples_{str(ii).zfill(3)}.npy", z_sample.detach().cpu().numpy())

    def save_model_gen_params(self, generator, models, optimizer, epoch, num_imgs, folder, sup_folder, model_type):
        torch.save(generator.state_dict(), f"./{sup_folder}/{folder}/model_checkpoints/cotrain-generator_{epoch.zfill(7)}.pt")
        torch.save(optimizer.state_dict(), f"./{sup_folder}/{folder}/model_checkpoints/optimizer_{epoch.zfill(7)}.pt")
        for mm in range(num_imgs):
            if model_type == "gmm" or model_type == "gmm_eye":
                mu, L = models[mm]
                np.savez(f'./{sup_folder}/{folder}/model_checkpoints/gmm-{mm}',
                              mu=mu.detach().cpu().numpy(), L=L.detach().cpu().numpy())
            elif model_type == "gmm_low" or (model_type == "gmm_low_eye"):
                mu, L, ep = models[mm]
                np.savez(f'./{sup_folder}/{folder}/model_checkpoints/gmm-{mm}',
                              mu=mu.detach().cpu().numpy(), L=L.detach().cpu().numpy(), ep=ep.detach().cpu().numpy())
            elif model_type == "gmm_custom":
                mu, L = models[mm]
                np.savez(f'./{sup_folder}/{folder}/model_checkpoints/gmm-{mm}',
                              mu=mu.detach().cpu().numpy(), L=L.detach().cpu().numpy())

    
        
