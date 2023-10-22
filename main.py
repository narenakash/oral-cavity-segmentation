import wandb

from utils import get_config

config = get_config("config.yaml")


def main():

    run()

    
if __name__ == "__main__":
    
    project_name = 'opmd-oralcavity-segmentation'

    wandb.init(
        project=project_name,
        config=config,
        mode="disabled"    
    )

    wandb.config.update(config)

    wandb.finish()