import torch
import scripts.pipeline_tactile as pipeline_tactile
import scripts.pipeline_deepsdf as pipeline_deepsdf
import yaml
import config_files
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Demo to reconstruct objects using tactile-gym.
"""


def main(args):
    # Pipeline to collect touch data
    args["folder_touch_sdf"] = pipeline_tactile.main(args)

    # Pipeline to reconstruct object from touch data
    pipeline_deepsdf.main(args)


if __name__ == "__main__":
    cfg_path = os.path.join(
        os.path.dirname(config_files.__file__), "pipeline_tactile_deepsdf.yaml"
    )
    with open(cfg_path, "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    _ = main(cfg)