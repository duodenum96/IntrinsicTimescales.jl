# Usage with FieldTrip

This tutorial will show how to use IntrinsicTimescales.jl with FieldTrip. We won't go into the details of preprocessing, since it is out of the scope of our package. We'll use the example data from this [Fieldtrip tutorial](https://www.fieldtriptoolbox.org/workshop/madrid2019/tutorial_cleaning/). 

```matlab
% Add fieldtrip to the path and read data
restoredefaultpath
addpath /path/to/fieldtrip
ft_defaults

% Read data
subj = 'sub-22';
rootpath = '/path/to/workshop/data/madrid2019'; 

% Read data
cfg = [];
cfg.dataset    = [rootpath, '/tutorial_cleaning/single_subject_resting/' subj '_task-rest_run-3_eeg.vhdr'];
cfg.channel    = 'all';
cfg.demean     = 'yes';
cfg.detrend    = 'no';
cfg.reref      = 'yes';
cfg.refchannel = 'all';
cfg.refmethod  = 'avg';
data = ft_preprocessing(cfg);
```

Similar to MNE tutorial, we'll make 10 second trials and compute the ACF/PSD for each trial. 

```matlab
cfg = [];
cfg.length = 10;
cfg.overlap = 0;
data_segmented = ft_redefinetrial(cfg, data);
```

To save the data in .MAT format, we'll first extract the data from FieldTrip struct, then use `save` function. Let's also note the sampling rate.

```matlab
fs = data.fsample % 250.0

data_array = cat(3, data_segmented.trial{:});
savepath = '/path/to/save/data'
save(fullfile(savepath, 'data.mat'), 'data_array', 'fs')
```

Now we will switch to Julia. To read .MAT files, we will use the `MAT.jl` package.


```julia
using MAT
using IntrinsicTimescales

data_path = "/path/to/data.mat"
data_dict = matread(data_path)
data = data_dict["data_array"]
fs = data_dict["fs"]
```

We'll note the size of the data, and compute the INT metrics using the `acw` function.

```julia
println(size(data)) # channels x time x trials

results = acw(data, fs, acwtypes=[:acw50, :auc, :tau], parallel=true, dims=2, trial_dims=3, average_over_trials=true)
```

That's all. 