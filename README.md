# weightflow

## Setting up the Environment 

To setup with mamba:

```bash
mamba create -n weightflow python=3.10
mamba activate weightflow
pip install -r requirements.txt
```

Or install the packages in `requirements.txt` with any package manager. 

## Download MNIST MLP Weights

To load the weights for an ReLU MLP with dimensions 784-10-10, download the zip file at [this GDrive url](https://drive.google.com/file/d/1w7K8Qt4-LyCES9XNuKfCUIKddJC2k0k9/view?usp=sharing) and place the unzipped contents under the `data` folder. 

## Collecting Weights from Scratch

Running this script saves weights from `reps` SGD trajectories every `log-weights-iter` steps, after the first `ignore-epochs` epochs. 

The model is an MLP trained on MNIST with 784 input and 10 output dimensions, and one layer of `hidden-dim` dimensions. 

Set `save-path` to point to an h5 file; e.g. `data/mnist_weights.h5`.

```bash
python train_mnist.py \
    --save-path <save_path> \
    --batch-size 64 \
    --log-weights-iter 4 \
    --epochs 5 \
    --learning-rate 0.01 \
    --hidden-dim 32 \
    --reps 3 \
    --ignore-epochs 2
```

## Training the Flow

The `train_flow.ipynb` notebooks walks through the steps of loading the collected weights, training a flow in one of the three geometries, evaluating it, as well as estimating the likelihoods of the sampled weights. 

## Running SLURM Jobs

The `jobs/submit_runs.py` script can be used to automatically submit a number of SLURM jobs based on the configs in `jobs/setups`, such as 
```bash
python submit_runs.py --setup mnist-transformer
```
for the config `setups/mnist-transformer.yaml`. 
