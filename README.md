# Reflected Schrödinger Bridge Matching
If you use this code, please cite the associated paper: <https://arxiv.org/abs/2607.03626>


## How to run

Choose experiment (required):
```
> python train.py experiment=afhq64
```
or, using accelerate to distribute on `N` GPUs,
```
> accelerate launch --num_processes N train.py experiment=...
```

For reflected Brownian bridge, add
```
> ... sde=reflected
```

Resume checkpoint with
```
> ... training.resume.dir=outputs/path/to/checkpoints/c_{final, 1000, etc.}
```

Evaluate checkpoint:
```
> python eval.py outputs/path/to/checkpoints/c_{final, 1000, etc.}
```
By default, full eval is run when training completes.
The eval will generate a cache of synthetic images.
When an experiment has been run with and without reflection, a side-by-side comparison can be performed using these caches:
```
> python compare.py path/to/non-reflected/cache path/to/reflected/cache
```


## Data
Instructions for retrieving AFHQ data is found at <https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq>.
Place under `data/` as follows.
```text
data/
└── AFHQ/
    ├── test/
    │   ├── cat/
    │   ├── dog/
    │   └── wild/
    └── train/
        ├── cat/
        ├── dog/
        └── wild/
```
`MNIST` and `EMNIST` should automatically download to the proper place. 


## Apptainer
The experiments were run using apptainer. We provide the definition file `my-pytorch.def`.

If running torch.compile inside apptainer, set the `TRITON_LIBCUDA_PATH` variable:
```
> apptainer exec --nv --env TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real accelerate launch ...
```
This is required for the `afhq64` experiment with the current config, as the EMA model is compiled to improve sampling speed during the finetuning stage.
To run without compiling the EMA model, remove `compile_ema: true` (or set it to false) in the training config located in `conf/training/`.
