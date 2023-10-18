
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