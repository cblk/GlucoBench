# GlucoBench

The official implementation of the paper "GlucoBench: Curated List of Continuous Glucose Monitoring Datasets with Prediction Benchmarks."
If you found our work interesting and plan to re-use the code, please cite us as:
```
@article{
  author  = {Renat Sergazinov and Valeriya Rogovchenko and Elizabeth Chun and Nathaniel Fernandes and Irina Gaynanova},
  title   = {GlucoBench: Curated List of Continuous Glucose Monitoring Datasets with Prediction Benchmarks},
  journal = {arXiv}
  year    = {2023},
}
```

# Dependencies

We recommend to setup clean Python environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env") by running `conda create -n glucobench python=3.10`. Then we can install all dependenices by running `pip install -r requirments.txt`. 

To run [Latent ODE](https://github.com/YuliaRubanova/latent_ode) model, install [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq).

# Code organization

The code is organized as follows:

- `bin/`: training commands for all models
- `config/`: configuration files for all datasets
- `data_formatter/`
    - base.py: performs **all** pre-processing for all CGM datasets
- `exploratory_analysis/`: notebooks with processing steps for pulling the data and converting to `.csv` files
- `lib/`
    - `gluformer/`: model implementation
    - `latent_ode/`: model implementation
    - `*.py`: hyper-paraemter tuning, training, validation, and testing scripts
- `output/`: hyper-parameter optimization and testing logs
- `paper_results/`: code for producing tables and plots, found in the paper
- `utils/`: helper functions for model training and testing
- `raw_data.zip`: web-pulled CGM data (processed using `exploratory_analysis`)
- `environment.yml`: conda environment file

# Data

The datasets are distributed according to the following licences and can be downloaded from the following links outlined in the table below.

| Dataset | License | Number of patients | CGM Frequency |
| ------- | ------- | ------------------ | ------------- |
| [Colas](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0225817#sec018) | [Creative Commons 4.0](https://creativecommons.org/licenses/by/3.0/us/) | 208 | 5 minutes |
| [Dubosson](https://doi.org/10.5281/zenodo.1421615) | [Creative Commons 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode) | 9 | 5 minutes |
| [Hall](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2005143#pbio.2005143.s010) | [Creative Commons 4.0](https://creativecommons.org/licenses/by/4.0/) | 57 | 5 minutes |
| [Broll](https://github.com/irinagain/iglu) | [GPL-2](https://www.r-project.org/Licenses/GPL-2) | 5 | 5 minutes |
| [Weinstock](https://public.jaeb.org/dataset/537) | [Creative Commons 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode) | 200 | 5 minutes |

To process the data, follow the instructions in the `exploratory_analysis/` folder. Processed datasets should be saved in the `raw_data/` folder. We provide examples in the `raw_data.zip` file.

## External research datasets

Additional datasets collected for the hidden baseline insulin and CGM dynamics research are staged under `output/external_datasets/raw/`. They are kept separate from the original GlucoBench benchmark datasets above: they are not bundled into `raw_data.zip`, are not loaded automatically by `DataFormatter`, and are not used by the training scripts unless a separate, approved validation workflow explicitly adds them.

| Dataset | Local status | Available material |
| ------- | ------------ | ------------------ |
| Kobe CGM_AC | Complete public download | Zenodo source data and CGM files, plus a GitHub repository snapshot |
| Stanford metabolic subphenotype | Complete public download | Public metabolic phenotype data package linked by the study |
| ShanghaiT1DM / ShanghaiT2DM | Complete public download | Figshare package containing CGM, clinical, laboratory, medication, and dietary records |
| Healthy non-diabetic reference (Dryad) | Public artifacts complete | Two versioned supplemental DOCX files; participant-level raw CGM is not exposed by Dryad |
| BIG IDEAs PhysioNet v1.1.3 | CGM research subset complete | All 16 Dexcom files, 16 food logs, demographics, license, and official checksums; the incomplete full multimodal ZIP is isolated and must not be used |

Completed external files are marked read-only. Unfinished downloads are stored only under an explicitly named `incomplete_*` directory and must never be treated as valid data. AI-READI, Human Phenotype Project / 10K, PREDICT 1, Framingham Exam 4, JAEB raw CGM, Singapore, and T2Help remain access-controlled, form-gated, or unpublished; a visible landing page does not mean that participant-level data has been acquired.

See the [external dataset landscape](reports/dataset_landscape_20260811.md) for candidate selection and validation roles, and the [download audit](reports/dataset_download_audit_20260811.md) for exact files, sizes, hashes, licenses, and access blockers. External data must remain separate from product changes: downloading or validating a dataset does not authorize changes to `index.html`, screening formulas, or thresholds.

# How to reproduce results?

## Setting up the enviroment

We recommend setting up a clean Python environment using [Conda](https://docs.conda.io/). Follow these steps:

1. Create a new environment named `glucobench` with Python 3.10 by running:
   ```
   conda env create -n glucobench python=3.10
   ```

2. Activate the environment with:
   ```
   conda activate glucobench
   ```

3. Install all required dependencies by running:
   ```
   pip install -r requirements.txt
   ```

4. (Optional) To confirm that you're installing in the correct environment, run:
   ```
   which pip
   ```
   This should display the path to the `pip` executable within the `glucobench` environment."

## Changing the configs

The `config/` folder stores the best hyper-parameters (selected by [Optuna](https://optuna.org)) for each dataset and model. The `config/` also stores the dataset-specific parameters for interpolation, dropping, splitting, and scaling. To train and evaluate the models with these defaults, we can simply run: 

```python
python ./lib/model.py --dataset dataset --use_covs False --optuna False
``` 

## Changing the hyper-parameters

To change the search grid for hyper-parameters, we need to modify the `./lib/model.py` file. Specifically, we look at the `objective()` function and modify the `trial.suggest_*` parameters to set the desired ranges. Once we are done, we can run the following command to re-run the hyper-parameter optimization:

```python
python ./lib/model.py --dataset dataset --use_covs False --optuna True
```

# How to work with the repository?

We provide a detailed example of the workflow in the `example.ipynb` notebook. For clarification, we provide some general suggestions below in order of increasing complexity. 

## Just the data
To start experimenting with the data, we can run the following command:

```python
import yaml
from data_formatter.base import DataFormatter

with open(f'./config/{dataset}.yaml', 'r') as f:
    config = yaml.safe_load(f)
formatter = DataFormatter(config)
```

The command exposes an object of class `DataFormatter` which automatically pre-processes the data upon initialization. The pre-processing steps can be controlled via the `config/` files. The `DataFormatter` object exposes the following attributes:

1. `formatter.train_data`: training data (as `pandas.DataFrame`)
2. `formatter.val_data`: validation data
3. `formatter.test_data`: testing (in-distribution and out-of-distribution) data
  i. `formatter.test_data.loc[~formatter.test_data.index.isin(formatter.test_idx_ood)]`: in-distribution testing data
  ii. `formatter.test_data.loc[formatter.test_data.index.isin(formatter.test_idx_ood)]`: out-of-distribution testing data
4. `formatter.data`: unscaled full data

## Integration with PyTorch

Training models with PyTorch typically boils down to (1) defining a `Dataset` class with `__getitem__()` method, (2) wrapping it into a `DataLoader`, (3) defining a `torch.nn.Module` class with `forward()` method that implements the model,  and (4) optimizing the model with `torch.optim` in a training loop. 

**Parts (1) and (2)** crucically depend on the definition of the `Dataset` class. Essentially, having the data in the table format (e.g. `formatter.train_data`), how do we sample input-output pairs and pass the covariate information? The various `Dataset` classes conveniently adopted from the `Darts` library (see [here](https://unit8co.github.io/darts/generated_api/darts.utils.data.training_dataset.html)) offer one way to wrap the data into a `Dataset` class. Different `Dataset` classes differ in what information is provided to the model:

1. `SamplingDatasetPast`: supports only past covariates
2. `SamplingDatasetDual`: supports only future-known covariates
3. `SamplingDatasetMixed`: supports both past and future-known covariates

Below we give an example of loading the data and wrapping it into a `Dataset`:

```python
from utils.darts_processing import load_data
from utils.darts_dataset import SamplingDatasetDual

formatter, series, scalers = load_data(seed=0,
                                       dataset=dataset,
                                       use_covs=True, 
                                       cov_type='dual',
                                       use_static_covs=True)
dataset_train = SamplingDatasetDual(series['train']['target'],
                                    series['train']['future'],
                                    output_chunk_length=out_len,
                                    input_chunk_length=in_len,
                                    use_static_covariates=True,
                                    max_samples_per_ts=max_samples_per_ts,)
```

**Parts (3) and (4)** are model-specific, so we omit their discussion. For inspiration, we suggest to take a look at the `lib/gluformer/model.py` and `lib/latent_ode/trainer_glunet.py` files.




