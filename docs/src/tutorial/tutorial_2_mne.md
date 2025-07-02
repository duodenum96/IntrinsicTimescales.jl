# Usage with MNE-Python

In this tutorial, we will give an example of using MNE-Python with IntrinsicTimescales.jl. 

We will start with downloading an [example resting state dataset](https://mne.tools/stable/documentation/datasets.html#resting-state). The details about the dataset can be found [here](https://neuroimage.usc.edu/brainstorm/DatasetResting). We won't detail the preprocessing here, MNE (as well as other packages) already have excellent tutorials on it. 

The first part is in python. Let's download the data and read it. 

```python
import mne
import mne
import os.path as op
import numpy as np

# Download example data and specify the path
data_path = mne.datasets.brainstorm.bst_resting.data_path()
raw_fname = op.join(data_path, "MEG", "bst_resting", "subj002_spontaneous_20111102_01_AUX.ds")

# Read data
raw = mne.io.read_raw_ctf(raw_fname)
```

We will epoch the data into 10 second epochs, from each of these epochs, we will compute one ACF/PSD. Then, we'll put the data into a numpy array.

```python
epochs = mne.make_fixed_length_epochs(raw, 10)
data = epochs.get_data()
```

Finally, we will save the data to computer so that we can read it in Julia. We'll also note the sampling rate.

```python
save_path = <path to save the data>
np.save(op.join(save_path, "data.npy"), data)
sampling_rate = raw.info['sfreq']
# 2400.0
```

Now we will switch to Julia. To read .npy files, we can use the `npzread` function from the `NPZ` package. Let's note the size of the data.

```julia
using IntrinsicTimescales
using NPZ

fs = 2400.0 # Sampling rate

data_path = <path to the data>
data = npzread(joinpath(data_path, "data.npy"))
println(size(data))
```

The data is in the shape of (n_epochs, n_channels, n_samples). Let's compute the INT metrics using the `acw` function. We'll average over the trials dimension (epochs in the MNE jargon) and use parallel processing.

```julia
results = acw(data, fs, acwtypes=[:acw50, :tau], dims=3, trial_dims=1, parallel=true, average_over_trials=true)
```

That's it. Note that you can also use [`PythonCall.jl`](https://github.com/JuliaPy/PythonCall.jl) to use MNE inside Julia. 