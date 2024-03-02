# Streamlining Speech Enhancement DNNs: an Automated Pruning Method Based on Dependency Graph with Advanced Regularized Loss Strategies

## Notice
This source code is mainly compatible for DF2 model. `model.py` can convert the current model to be pruned and the modules of dataloader and validation in `train.py` and `pruning.py` can also be replaced with your preference.

## Training with Sparsity

### Sparsity Loss
the main sparsity losses are in `Loss.py`.

### DF2 Model

Install the DeepFilterNet Python wheel via pip:
```bash
# Install cpu/cuda pytorch (>=1.9) dependency from pytorch.org, e.g.:
pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
# Install DeepFilterNet
pip install deepfilternet
# Or install DeepFilterNet including data loading functionality for training (Linux only)
pip install deepfilternet[train]
```

The detail can be found in https://github.com/Rikorose/DeepFilterNet.

### DF2 Data Preparation

The entry point is `train.py`. It expects a data directory containing HDF5 dataset
as well as a dataset configuration json file.

So, you first need to create your datasets in HDF5 format. Each dataset typically only
holds training, validation, or test set of noise, speech or RIRs.
```py
# 1.Install additional dependencies for dataset creation
pip install h5py librosa soundfile
# 2.Prepare text file (e.g. called training_set.txt) containing paths to .wav files
#
# usage: prepare_data.py [-h] [--num_workers NUM_WORKERS] [--max_freq MAX_FREQ] [--sr SR] [--dtype DTYPE]
#                        [--codec CODEC] [--mono] [--compression COMPRESSION]
#                        type audio_files hdf5_db
#
# where:
#   type: One of `speech`, `noise`, `rir`
#   audio_files: Text file containing paths to audio files to include in the dataset
#   hdf5_db: Output HDF5 dataset.
python df/scripts/prepare_data.py --sr 48000 speech training_set.txt TRAIN_SET_SPEECH.hdf5
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
pip install torch-pruning 
```
The detail can be found in https://github.com/VainF/Torch-Pruning.

### Pruning for DF2
The enter point is `pruning.py` and `prune_utils.py` is the corresponding dependency package.You may change the arguments with detailed explanation in `pruning.py`.
```py
# usage: pruning.py 
python pruning.py
```

### Pruning for others

Substitute the validation and Dataloader module to your own implementation. 

## Evaluation

The enter point is `test_dataset.py`, all results of performance evaluation in the paper are derived through it.You may change the arguments with detailed explanation in `test_dataset.py`.
```py
# usage: test_dataset.py 
python test_dataset.py
```