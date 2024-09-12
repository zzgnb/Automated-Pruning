# Streamlining Speech Enhancement DNNs: an Automated Pruning Method Based on Dependency Graph with Advanced Regularized Loss Strategies

## Result Preview
### DeepFilterNet2 48k/16k 
[DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio](https://github.com/Rikorose/DeepFilterNet)
<img width="827" alt="image" src="https://github.com/user-attachments/assets/60dc4c2c-a688-4cb7-9f90-a1a06c85c171">
-p means pruned model, -small means the version with fewer params.
### DeepVQE
[DeepVQE: Real Time Deep Voice Quality Enhancement for Joint Acoustic Echo Cancellation, Noise Suppression and Dereverberation](https://arxiv.org/pdf/2306.03177).
<img width="792" alt="image" src="https://github.com/user-attachments/assets/146d91da-ac41-45be-9585-bf0acd3b4c55">

## Notice
This source code is mainly compatible for DF2 model. `model.py` can convert the current model to be pruned and the modules of dataloader and validation in `train.py` and `pruning.py` can also be replaced with your preference.

## Training with Sparsity

### Sparsity Loss
the main sparsity losses are in `Loss.py`. Actually, purning is an empirical process so the various factor settings of sparsity loss should be tried for your own model pruning. Limited by the sub-optimal analysis of weight significance(L1Norm we use), pruning remains indeterminacy and we suggest to pruning more than once or test more effective analysis methods.

### DF2 Model

Install the DeepFilterNet Python wheel via pip:
```bash
# Install DeepFilterNet
pip install deepfilternet
# Or install DeepFilterNet including data loading functionality for training (Linux only)
pip install deepfilternet[train]
```

The detail can be found in https://github.com/Rikorose/DeepFilterNet.

### Data Preparation for DF2 Pruning

The entry point is `train.py`. It expects a data directory containing HDF5 dataset
as well as a dataset configuration json file.

So, you first need to create your datasets in HDF5 format. Each dataset typically only
holds training, validation, or test set of noise, speech or RIRs.
```py
# 1.Install dependencies 
pip install h5py librosa soundfile
# 2.Prepare text file (e.g. called training_set.txt) containing paths to .wav files
python scripts/get_txt.py
# 3.Generate HDF5 files
python scripts/prepare_data.py [--num_workers NUM_WORKERS] [--sr SR] [--dtype DTYPE] [path.txt] [data.hdf5]
# where:
#   type: One of `speech`, `noise`, `rir`
#   audio_files: Text file containing paths to audio files to include in the dataset
#   hdf5_db: Output HDF5 dataset.
# for example: python scripts/prepare_data.py --sr 48000 speech training_set.txt TRAIN_SET_SPEECH.hdf5
```
All datasets should be made available in one dataset folder for the train script.

The dataset configuration file should contain 3 entries: "train", "valid", "test". Each of those
contains a list of datasets (e.g. a speech, noise and a RIR dataset). You can use multiple speech
or noise dataset. Optionally, a sampling factor may be specified that can be used to over/under-sample
the dataset. Say, you have a specific dataset with transient noises and want to increase the amount
of non-stationary noises by oversampling. In most cases you want to set this factor to 1.

<details>
  <summary>Dataset config example:</summary>
<p>
  
`dataset.cfg`

```json
{
  "train": [
    [
      "TRAIN_SET_SPEECH.hdf5",
      1.0
    ],
    [
      "TRAIN_SET_NOISE.hdf5",
      1.0
    ],
    [
      "TRAIN_SET_RIR.hdf5",
      1.0
    ]
  ],
  "valid": [
    [
      "VALID_SET_SPEECH.hdf5",
      1.0
    ],
    [
      "VALID_SET_NOISE.hdf5",
      1.0
    ],
    [
      "VALID_SET_RIR.hdf5",
      1.0
    ]
  ],
  "test": [
    [
      "TEST_SET_SPEECH.hdf5",
      1.0
    ],
    [
      "TEST_SET_NOISE.hdf5",
      1.0
    ],
    [
      "TEST_SET_RIR.hdf5",
      1.0
    ]
  ]
}
```

</p>
</details>

Finally, start the training script. The training script may create a model `base_dir` if not
existing used for logging, some audio samples, model checkpoints, and config. You may change the arguments with detailed explanation in `train.py`.
```py
# usage: train.py 
python train.py
```

## Automatic Pruning

### Environment Support 
Install the Torch-Pruning Python wheel via pip:
```bash
conda create -n [your_env_name] python=3.8
pip install torch-pruning==1.3.1
```
We modify these source files in directory `source` and you could substitute these in vitual environment for better compatability. The corresponding paths are:
```bash
/your_path_to_anaconda/anaconda3/envs/Speech/lib/python3.8/site-packages/torch_pruning/pruner/function.py
/your_path_to_anaconda/anaconda3/envs/Speech/lib/python3.8/site-packages/torch_pruning/dependency.py
/your_path_to_anaconda/anaconda3/envs/Speech/lib/python3.8/site-packages/torch_pruning/ops.py
```
The detail can be found in https://github.com/VainF/Torch-Pruning. By the way, Torch-Pruning is not that compatible and intelligent and there is need to implement Pruner for specific network layers such as grouped linear layers for DF2, named `GLinearPruner` in `prune_utils.py`.  

### Pruning for DF2
The enter point is `pruning.py` and `prune_utils.py` is the corresponding dependency package.You may change the arguments with detailed explanation in `pruning.py`.
```py
# usage: pruning.py 
python pruning.py
```

### Pruning for others

Substitute the validation and Dataloader module to your own implementation in `pruning.py`.

## Evaluation

The enter point is `test_dataset.py`, all results of performance evaluation in the paper are derived through it.You may change the arguments with detailed explanation in `test_dataset.py`.
```py
# usage: test_dataset.py 
python test_dataset.py
```
