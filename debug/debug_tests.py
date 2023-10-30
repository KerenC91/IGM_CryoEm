import debug
import yaml

# Plot losses comparison test
loss_names = ["loss", "loss_data", "loss_ent", "loss_prior"]
comparison_params = ["num_imgs", "latent_dim", "sigma", "latent_type"]
comparison_details_dict = {
    "num_imgs": ['5', '35', '75', '150'],
    "latent_dim": ['8', '40', '150'],
    "sigma": ['0.5', '1.0', '2.0'],
    "latent_type": ['gmm', 'gmm_custom', 'gmm_eye']
}
# Compare
for comparison_param in comparison_params:
    comparison_details = comparison_details_dict[comparison_param]
    for loss in loss_names:
        output_folder = "../results_batched/comparison"
        output_file = f"{loss}_comparison_{comparison_param}"
        folders = debug.get_folders_list(comparison_param, comparison_details)
        comparison_details_string = ",".join(comparison_details)
        debug.plot_losses_comparison(folders,
                                     base_folder="../results_batched",
                                     loss_file=f"{loss}.npy",
                                     output_folder=output_folder,
                                     output_file=output_file,
                                     diff=f"Compare {comparison_param}: {comparison_details_string}. {loss}",
                                     jump_size=100,
                                     clip_epochs=False,
                                     comparison_details=comparison_details)

    # Write the comparison details to a YAML file
    debug.write_comparison_args(output_folder, output_file, comparison_param, comparison_details, folders)

