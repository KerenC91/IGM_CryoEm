import debug


# Plot losses comparison test
debug.plot_losses_comparison("../results_batched/MNIST648_gmm_denoising_deepdecoder_5imgs_0.5noise_std_dropout0.0001_layer_size150x6_latent8_seed100_eps0.001/loss.npy",
                             "../results_batched/MNIST648_gmm_denoising_deepdecoder_5imgs_0.5noise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001/loss.npy",
                             "../results_batched/MNIST648_gmm_denoising_deepdecoder_35imgs_0.5noise_std_dropout0.0001_layer_size150x6_latent40_seed100_eps0.001/loss.npy",
                             "../results_batched/MNIST648_gmm_denoising_deepdecoder_50imgs_0.5noise_std_dropout0.0001_layer_size150x6_latent8_seed100_eps0.001/loss.npy",
                             output_folder="../results_batched/comparison")
                             #diff="num_imgs:5, 5, 35, 50. latent_dim:8, 40, 40, 8")