# utils.py
import yaml

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def create_directories(config, load_checkpoint=None):
    """
    Create or load directories for saving model outputs.
    
    Parameters:
        config (dict): Configuration dictionary containing paths
        load_checkpoint (str, optional): Path to checkpoint file to load from
        
    Returns:
        tuple: Paths to save_dir, checkpoint_dir, plot_dir, and tensorboard_dir
    """
    import os
    
    if load_checkpoint:
        # Extract directory path from checkpoint path
        checkpoint_dir = os.path.dirname(load_checkpoint)
        save_dir = os.path.dirname(checkpoint_dir)
        plot_dir = os.path.join(save_dir, "plots")
        tensorboard_dir = os.path.join(save_dir, "tensorboard")
    else:
        # Create new directories with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(config["paths"]["save_dir"], timestamp)
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        plot_dir = os.path.join(save_dir, "plots")
        tensorboard_dir = os.path.join(save_dir, "tensorboard")
        
        # Create all directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # Save config for new runs
        # import json
        # with open(os.path.join(save_dir, "config.json"), "w") as f:
        #     json.dump(config, f, indent=4)

    return save_dir, checkpoint_dir, plot_dir, tensorboard_dir
