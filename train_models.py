import os

#os.system(f"glupredkit calculate_metrics --metrics rmse --models tcn__559_insulin__30.pkl,tcn__563_insulin__30.pkl,tcn__570_insulin__30.pkl,tcn__575_insulin__30.pkl,tcn__588_insulin__30.pkl,tcn__591_insulin__30.pkl")
#os.system(f"glupredkit calculate_metrics --metrics rmse --models lstm__559_insulin__30.pkl,lstm__563_insulin__30.pkl,lstm__570_insulin__30.pkl,lstm__575_insulin__30.pkl,lstm__588_insulin__30.pkl,lstm__591_insulin__30.pkl")



configurations = ["_cgm", "_all"]

model_file_names = ""

for person in ['559', '563', '570', '575', '588', '591']:

    """
    # Parse data
    os.system(f"python -m glupredkit.cli parse --parser ohio_t1dm --file-path data/raw/ --subject-id {person} --output-file-name {person}")
    """

    # Train models
    for config in configurations:
        config_name = f"access_{person}{config}"
        #os.system(f"python -m glupredkit.cli train_model ridge {config_name}")
        model_file_names = model_file_names + f"ridge__{config_name}__30.pkl," + f"ridge__{config_name}__60.pkl,"

me_model_file_names = ""
for config in configurations:
    #os.system(f"python -m glupredkit.cli train_model ridge access_me{config}")
    me_model_file_names = (me_model_file_names + f"ridge__access_me{config}__30.pkl,"
                           + f"ridge__access_me{config}__60.pkl,")


model_file_names = model_file_names[:-1]
me_model_file_names = me_model_file_names[:-1]
#os.system(f"python -m glupredkit.cli calculate_metrics --metrics rmse,mre --models {model_file_names}")
os.system(f"python -m glupredkit.cli calculate_metrics --metrics rmse,mre --models {me_model_file_names}")
#os.system(f"python -m glupredkit.cli calculate_metrics --metrics parkes_error_grid --models {me_model_file_names}")


#os.system(f"python -m glupredkit.cli draw_plots --models {me_model_file_names} --plots trajectories --start-date 04-12-2022/15:00 --end-date 05-12-2022/15:00")
#os.system(f"python -m glupredkit.cli draw_plots --models ridge__access_me_cgm__30.pkl,ridge__access_me_all__30.pkl --plots scatter_plot")
#os.system(f"python -m glupredkit.cli draw_plots --models ridge__access_me_cgm__30.pkl --plots scatter_plot")
#os.system(f"python -m glupredkit.cli draw_plots --models ridge__access_me_all__30.pkl --plots scatter_plot")
#os.system(f"python -m glupredkit.cli draw_plots --models ridge__access_me_cgm__60.pkl,ridge__access_me_all__60.pkl --plots scatter_plot")
