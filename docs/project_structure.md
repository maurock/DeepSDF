# Project structure

This markdown describes the structure of this project.

- `examples/reconstruct_single.py`: reconstruct single shape using thr DeepSDF model.
- `examples/reconstruct_custom_from_latent.py`: reconstruct a shape from an optimised latent vector. This script requires model weights and a torch tensor of the optimised latent vector.
- `examples/reconstruct_all_from_latent.py`: reconstruct all the shapes from the latent vectors optimised during training.
- `examples/infer_shape.py`: infer a custom shape using an optimised network. In this script, we first sample points on the object surface and then infer its entire shape.
