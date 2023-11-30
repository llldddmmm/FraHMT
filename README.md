# Code for the paper "FraHMT: A fragment-oriented heterogeneous graph molecular generation model for target proteins"

### ZINC dataset
Since our dataset exceeds the maximum upload capacity of Github, please download ZINC dataset at this link(https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv). 
After downloading, store it in the RAW folder under the DATA folder.

### Preprocessing
First, you need to download the data and do some preprocessing. To do this, run:

`python manage.py preprocess --dataset <DATASET_NAME>`

where `<DATASET_NAME>` must be `ZINC` or `EGFR` or `HTR1A`. 

### Training
After preprocessing, you can train the model running:

`python manage.py train --dataset ZINC`

If you wish to train using a GPU, add the `--use_gpu` option.
Training the model will create folder `RUNS` with the following structure:
```
RUNS
└── <date>@<time>-<hostname>-<dataset>
    ├── ckpt
    │   ├── best_loss.pt
    │   ├── best_valid.pt
    │   └── last.pt
    ├── config
    │   ├── config.pkl
    │   ├── emb_<embedding_dim>.dat
    │   ├── params.json
    │   └── vocab.pkl
    ├── results
    │   ├── performance
    │   │   ├── loss.csv
    │   │   └── scores.csv
    │   └── samples
    └── tb
        └── events.out.tfevents.<tensorboard_id>.<hostname>
```
the `<date>@<time>-<hostname>-<dataset>` folder is a snapshot of your experiment, which will contain all the data collected during training.
You can monitor the progress of training using tensorboardX, just run

`tensorboard --logdir RUNS`

during training and check the `localhost:6006` page in your favorite browser.

### Training_transfer
After pre-training, you can perform transfer training:

`python manage.py train_transfer --dataset <DATASET_NAME>`

where `<DATASET_NAME>` must be `EGFR` or `HTR1A`. 
If you wish to train using a GPU, add the `--use_gpu` option.

Here you need to provide a `transfer_run_dir`, which is the path to the pre-training running directory.

### Sampling
After the model is trained, you can sample from it using

`python manage.py sample --run <RUN_PATH>`

where `<RUN_PATH>` is the path to the run directory of the experiment, which will be something like `RUNS/<date>@<time>-<hostname>-<dataset>` (`<date>`, `<time>`, `<hostname>`, `<dataset>` are placeholders of the actual data).
You will find your samples in the `results/samples` folder on your experiment run directory.

### Postprocessing
After you have sampled the model, you wish to conduct some common postprocessing operations such as calculate statistics on the samples, aggregate multiple sample files and the test data in one big file for plotting, etc.
Then, you need to run:

`python manage.y postprocess --run <RUN_PATH>`

where `<RUN_PATH>` is obtained as described above.

### Plotting
If you wish to obtain similar figures as the ones in the paper on your samples, just run:

`python manage.py plot --run <RUN_PATH>`

