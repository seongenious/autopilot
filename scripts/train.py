"""Training script with Hydra configuration."""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object.
    """
    print(OmegaConf.to_yaml(cfg))

    # TODO: Implement training logic
    # 1. Build model from cfg.model
    # 2. Build dataset from cfg.dataset
    # 3. Build trainer from cfg.training
    # 4. Train


if __name__ == "__main__":
    main()
