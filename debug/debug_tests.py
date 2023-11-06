import debug

# Plot losses comparison test

debug_json_config_path = "debug_config.json"
args = debug.parse_config_file(debug_json_config_path)

# Compare
for comparison_param in args["comparison_params"]:
    comparison_details = args["comparison_details_dict"][comparison_param]
    output_folder = "../results_batched/comparison"
    for loss_name in args["loss_names"]:

        output_file = f"{loss_name}_comparison_{comparison_param}"
        folders = debug.get_folders_list(comparison_param, comparison_details)
        comparison_details_string = ",".join(comparison_details)
        debug.plot_losses_comparison(folders,
                                     base_folder="../results_batched",
                                     loss_name=loss_name,
                                     output_folder=output_folder,
                                     output_file=output_file,
                                     diff=f"Compare {comparison_param}: {comparison_details_string}. {loss_name}",
                                     jump_size=100,
                                     clip_epochs=False,
                                     comparison_details=comparison_details)

    # Write the comparison details to a YAML file
    debug.write_comparison_args(output_folder, comparison_param, comparison_details, folders)

