# CAISR GitHub Readme

## Introduction
The **CAISR system** is designed to streamline tasks related to sleep data analysis by using Docker containers for ease of use. The primary method of using this system is by downloading pre-built Docker images from our website. However, the Python code used to create these images is also available here for transparency and customization purposes. Users who wish to adapt the system to their own datasets or analysis preferences can rebuild the Docker images with minimal effort.

> **New:** every CAISR module — `caisr_stage.py`, `caisr_arousal.py`, `caisr_resp.py`, `caisr_limb.py` — can be run **natively in Conda, no Docker required**, and now accepts **either pre-processed `.h5` files or raw `.edf` recordings in the same input directory**. Input format is auto-detected per file by extension; the same script processes both. See the [Running CAISR Natively (No Docker)](#running-caisr-natively-no-docker) section.

## Prerequisites
- **Docker**: Ensure Docker is installed on your machine. You can download it [here](https://www.docker.com/products/docker-desktop).
- **Python**: Python 3.7+ should be installed.
- **Required Libraries**: Ensure you have the following Python libraries installed: `docker` and `subprocess`.

  Install them using:
  ```bash
  pip install docker subprocess
  ```

## Directory Structure
The CAISR system requires a specific directory structure to function correctly. Here’s how you should organize your files and folders:

```
your_project_directory/
│
├── dockers/                 # Contains all the Docker image tar.gz files
│   ├── caisr_preprocess.tar.gz
│   ├── caisr_stage.tar.gz
│   ├── caisr_arousal.tar.gz
│   ├── caisr_resp.tar.gz
│   ├── caisr_limb.tar.gz
│   └── caisr_report.tar.gz
│
├── data/raw/                # Input data folder that will be processed
│   ├── file1.edf
│   ├── file2.edf
│   └── ... (other files)
│
├── data/                    # H5 files after basic preprocessing/resampling
│   ├── file1.h5             
│   ├── file2.h5
│
├── caisr_output/            # Output folder where results will be stored
│   ├── stage/
│   ├── arousal/
│   ├── resp/
│   ├── limb/
│   └── report/
│
└── caisr.py                 # The main script provided in this readme
```

## Setup

### 1. **Prepare the Docker Images**:
   - Place the `*.tar.gz` Docker files in the `dockers/` folder.
   - These will be loaded automatically by the script if not already installed.

### 2. **Configure Input and Output Folders**:
   - Place your input data files (e.g., `.edf` files) in the `data/raw/` folder.
   - The script will automatically create necessary subfolders in the `caisr_output/` folder based on the tasks you run.

### 3. **Expected Input Data**:

You can either supply `.edf` files or `.h5` files.

#### **EDF Files:**

CAISR expects a certain minimum number of channels to be available. The algorithm accounts for some basic channel renaming and imputation of missing channels. However, it’s necessary to follow the channel naming conventions below for optimal performance:

- **EEG (Electroencephalogram)**:
  - At least one EEG is required, named either:  
    `f3-m2`, `f4-m1`, `c3-m2`, `c4-m1`, `o1-m2`, `o2-m1`, or `eeg`.
  - CAISR performs best when six EEG channels are provided:  
    `f3-m2`, `f4-m1`, `c3-m2`, `c4-m1`, `o1-m2`, `o2-m1`.

- **EOG (Electrooculogram)**:
  - At least one EOG is required, named either:  
    `e2-m1` or `e1-m2`.

- **Chin EMG (Electromyogram)**:
  - A chin EMG channel `chin1-chin2` is required.

- **Respiratory Channels**:
  - Either `ptaf` (pressure transducer airflow) or `cflow` (flow measured during CPAP (Continuous Positive Airway Pressure)) is required. If both are set to zero, `abd` + `chest` will be used as the primary breathing trace. 
  - Both abdominal and thoracic effort belt signals are required, named:  
    `abd` (abdominal) and `chest` (thoracic).
  - Oxygen saturation signal is required, named:  
    `spo2`.

  - **Optional Respiratory Channels for Optimal Performance**:
    - `airflow`, `cflow`, `cpap_on`.  
      Note: `airflow` measured via thermistor is ideal, yet optional, when `ptaf` or `cflow` is provided. `ptaf` or `cflow` should be set to all zeros unless it's a titration/split-night study. In the case of a titration/split-night study, ideally a binary indicator `cpap_on` (e.g., 0000001111111) is provided.

- **Leg EMG (Electromyogram)**:
  - Two leg EMGs are required for limb movement analysis, named:  
    `lat` (left leg) and `rat` (right leg).

- **Optional**:
  - Heart rate channel: `hr`.
  - Body position channel: `position` (used in the report but not in the analysis).

- **Sampling Rate**:  
  Any sampling rate can be provided, as CAISR will resample the data to **200 Hz**.

#### **H5 Files:**

You can provide `.h5` files instead of `.edf` files. This skips the channel renaming logic that occurs with `.edf` files. Therefore, the `.h5` files need to match the required format exactly.

- **Expected Sampling Rate**: 200 Hz.
- **Required Channels**: 
  ```
  'f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1',
  'e1-m2', 'e2-m1', 'chin1-chin2', 'abd', 'chest', 'spo2', 'ecg', 'lat', 'rat'
  ```
- **Optional Channels**: 
  ```
  'airflow', 'cpap_on', 'hr', 'position'
  ```

The Python code to create a suitable `.h5` file is:

```python
def save_prepared_data(path, signal):
    with h5py.File(path, 'w') as f:
        f.attrs['sampling_rate'] = 200
        f.attrs['unit_voltage'] = 'V'
        group_signals = f.create_group('signals')
        for name in signal.columns:
            group_signals.create_dataset(name, data=signal[name], shape=(len(signal), 1), 
                                         maxshape=(len(signal), 1), dtype='float32', compression="gzip")
```

### 4. **Specify the Tasks to Run**:
   - In the `caisr.py` file, modify the `tasks` list to include the tasks you want to execute.
   - Available tasks are:
     - `preprocess`: Preprocessing of raw data.
     - `stage`: Sleep stage classification.
     - `arousal`: Arousal detection.
     - `resp`: Respiratory analysis.
     - `limb`: Limb movement analysis.
     - `report`: Generate a comprehensive report based on the analysis.
   - Preprocessing is optional. If your data is already preprocessed, you can remove it from the list.

   ```python
   # Example of task configuration
   tasks = ['preprocess', 'stage', 'arousal', 'resp', 'limb', 'report']
   ```

### 5. **Run the Script**:

![image](https://github.com/user-attachments/assets/f081e3ce-12f1-41a2-91c4-4e3a3e838749)



   - Execute the main script using Python:
     ```bash
     python caisr.py
     ```
   - The script will automatically load Docker images, run the specified tasks, and output the results into the `caisr_output/` folder.

## Running CAISR Natively (No Docker)

Each of the four CAISR modules (`caisr_stage.py`, `caisr_arousal.py`, `caisr_resp.py`, `caisr_limb.py`) can be run directly from Python inside a Conda environment, with no Docker image required. The same script handles both pre-processed `.h5` files and raw `.edf` recordings — input format is detected per file from the extension, so a single input directory can mix the two if you want.

### How input dispatch works

* `*.h5` files are read as before, matching the [H5 File Specification](#h5-files) above.
* `*.edf` files are loaded via `edf_loader_pyedflib.py` (used by Stage / Resp / Limb) or `edf_loader.py` (used by Arousal). Both loaders handle channel renaming, bipolar derivations, per-channel resampling to 200 Hz, and unit conversion (signals are returned in Volts to match the H5 convention). Channel aliases live in `channel_table.csv` — to support a new EDF label, append it to the relevant alias row, no code changes needed.

The downstream model, post-processing, and CSV output format are identical regardless of input format.

### TS-Zarr input (BDSP primary stores, untested)

The native scripts also accept BDSP primary TS-Zarr stores
(`<subject>_<session>_task-psg_eeg.zarr` directories, the BDSP default
recording format) in the same input directory as `.h5` and `.edf` files. A
store is read like a raw EDF: `zarr_loader.load_zarr_for_caisr` returns the
same channel dictionary as `load_edf_for_caisr` (lower-case labels, voltage
channels in volts, names standardized with `channel_table.csv`, the same
bipolar derivations), and each module then resamples and prepares it exactly
as for EDF input.

This needs the shared BDSP reader in each task environment:
`pip install bdsp` (Python 3.11 or newer; it installs `zarr`). The Docker
images and the `caisr_preprocess.py` H5 preparation are unchanged.

Status: implemented and syntax-checked, and its signal reader was checked on a
synthetic store, but it has not been validated end to end against the CAISR
models. Compare its outputs with the EDF or H5 path before relying on it.

### 1. Create the Conda environments

Each module has its own dependency set, so create one environment per task:

```bash
# Sleep staging
cd stage && conda env create -f caisr_stage.yml && cd ..    # env: caisr_stage

# Arousal detection
conda create -n caisr_arousal python=3.9 -y
conda activate caisr_arousal
pip install -r arousal/arousal_requirements.txt
conda deactivate

# Respiratory analysis
conda create -n caisr_resp python=3.9 -y
conda activate caisr_resp
pip install -r resp/resp_requirements.txt
conda deactivate

# Limb movement
conda create -n caisr_limb python=3.9 -y
conda activate caisr_limb
pip install -r limb/limb_requirements.txt
conda deactivate
```

The EDF dependencies (`pyEDFlib` for stage / resp / limb, `edfio` for arousal) are already declared in each module's requirements file / env YAML, so the steps above install everything needed for both H5 and EDF input.

### 2. Run the modules

Activate the matching environment and run the script. Point `--input_data_dir` at a folder containing `.h5` files, `.edf` files, or a mix of both:

```bash
conda activate caisr_stage
python caisr_stage.py   --input_data_dir ./my_dataset --output_csv_dir ./caisr_output --model_dir ./stage/models --param_dir ./data/run_parameters

conda activate caisr_arousal
python caisr_arousal.py --input_data_dir ./my_dataset --output_csv_dir ./caisr_output --param_dir ./data/run_parameters

conda activate caisr_resp
python caisr_resp.py    --input_data_dir ./my_dataset --output_csv_dir ./caisr_output --param_dir ./data/run_parameters

conda activate caisr_limb
python caisr_limb.py    --input_data_dir ./my_dataset --output_csv_dir ./caisr_output --param_dir ./data/run_parameters
```

Per-subject CSVs are written to `./caisr_output/{task}/`. You can then aggregate them with the existing combine / report scripts exactly as in the Docker workflow.

---

## Customization: Adapting Preprocessing and Report Generation
While the CAISR system is optimized for standard `.edf` files and report formats, users with unique datasets or specific reporting needs can customize the preprocessing and reporting scripts. This flexibility allows you to handle non-standard `.edf` files or tailor the output reports to better suit your research requirements.

After making your desired changes to the preprocessing or report generation code, you can easily rebuild the Docker images by running:
```bash
python create_caisr_dockers.py
```
This automated script ensures that your customized Docker containers are quickly generated and ready for use.

## Script Workflow

### 1. **Listing Available Docker Images**
The script will first list all available Docker images on your system.

### 2. **Matching Docker Images to Tasks**
The script will match the Docker images to the specified tasks. If a required Docker image is missing, it will be automatically loaded from the `dockers/` folder.

### 3. **Running Tasks Inside Docker Containers**
For each task:
- The script will run the corresponding Docker container.
- It will mount the `data/` folder as input and `caisr_output/` folder as output.

### 4. **Output**
Results from each task will be stored in separate subfolders within the `caisr_output/` folder. Each subfolder corresponds to a specific task, such as `stage`, `arousal`, etc.

The numerical - per sample - output of CAISR is stored

 in `caisr_output/combined/`, containing the sleep stage hypnogram, the respiratory events, arousal events, and limb movement events. These output CSV files are saved at 2 Hz.

**Path:** `caisr_annotations/caisr_{study_id}.csv`

This file contains the per-sample time series output of CAISR.  
**One row = one timestamp**, and the file represents the **combined output** of all four CAISR sub-models:

- Sleep staging
- Arousal detection
- Respiratory event detection
- Limb movement detection

Sample Frequency: 2 Hz (one row = 0.5 seconds)

The saved output is **2 Hz**.

- Internal processing runs at **200 Hz**, and the final combined annotation file is **downsampled by taking every 100th sample**.  
  - `README.md:183-186`  
  - `caisr_report.py:177-178`

---

#### Columns and Their Coding

##### Index columns

| Column     | Description |
|-----------|-------------|
| `start_idx` | Start sample index in the **200 Hz** source `.h5` file |
| `end_idx`   | End sample index in the **200 Hz** source `.h5` file |

These columns are retained from the arousal task output and placed at the front of the combined file.  
- `caisr_report.py:153-175`

---

#### Sleep Stage columns (`stage`, `stage_prob_*`)

##### `stage` codes

| Value | Meaning |
|------:|---------|
| 1 | N3 (Deep Sleep) |
| 2 | N2 |
| 3 | N1 (Light Sleep) |
| 4 | REM |
| 5 | Wake |
| 9 | No stage (undefined / padded) |

##### Stage per-class probabilities (0.0–1.0)

The staging model also outputs per-class probabilities stored as floats in **[0.0, 1.0]**, rounded to **5 decimal places**:

| Column | Meaning |
|--------|---------|
| `stage_prob_n3` | Probability of N3 |
| `stage_prob_n2` | Probability of N2 |
| `stage_prob_n1` | Probability of N1 |
| `stage_prob_r`  | Probability of REM |
| `stage_prob_w`  | Probability of Wake |

---

#### Arousal columns (`arousal`, `arousal_prob_*`, `arousal_pp_trace`)

| Column | Values | Meaning |
|--------|--------|---------|
| `arousal` | 0 or 1 | 0 = no arousal, 1 = arousal event |
| `arousal_prob_no` | 0.0–1.0 | Model probability of no arousal |
| `arousal_prob_arousal` | 0.0–1.0 | Model probability of arousal |
| `arousal_pp_trace` | 0 or 1 | Post-processed arousal trace (binary) |

---

#### Respiratory event column (`resp`)

The `resp` column uses integer codes:

| Value | Meaning |
|------:|---------|
| 0 | No event |
| 1 | Obstructive Apnea (OA) |
| 2 | Central Apnea (CA) |
| 3 | Mixed Apnea (MA) |
| 4 | Hypopnea (HY) |
| 5 | Respiratory Effort-Related Arousal (RERA) |

This coding is confirmed in both the visualization code and AHI computation logic:  
- `caisr_report.py:676-686`  
- `sleep_indices.py:26-40`

---

#### Limb movement column (`limb`)

| Value | Meaning |
|------:|---------|
| 0 | No limb movement |
| 1 | Isolated limb movement |
| 2 | Periodic limb movement (PLM) |

Notes:
- The `prob_no` and `prob_limb` columns from the intermediate limb file are dropped before saving the combined output.
- The `plm` column is also dropped at this stage.  
  - `caisr_report.py:171`

The limb coding is also confirmed by the LMI/PLMI computation logic, which:
- treats values **1, 2, and 4** as limb movements
- treats **2** as periodic  
- `sleep_indices.py:122-127`

---

#### Missing data and storage rules

- Any `NaN` values in **any** column are replaced with **`9`** before saving.  
- **Non-probability columns** are stored as **integers**.  
- **Probability columns** are stored as **floats**, rounded to **5 decimal places**.  
  - `caisr_report.py:160-169`

---

### Report Generation

For each input data file, CAISR generates the following output:

- **PDF Report**:  
  A comprehensive report in `.pdf` format displaying polysomnography signals and CAISR analysis results.

- **CSV File**:  
  A `.csv` file (`caisr_sleep_metrics_all_studies.csv`) containing summary statistics from the sleep staging tasks, with the following key metrics:

    - **Total Sleep Metrics**:
      - `TST (h)`: Total Sleep Time in hours
      - `Recording (h)`: Total Recording Time in hours
      - `Eff (%)`: Sleep Efficiency percentage

    - **Sleep Stage Distribution**:
      - `REM (%)`: Percentage of time spent in REM sleep
      - `N1 (%)`: Percentage of time spent in N1 sleep
      - `N2 (%)`: Percentage of time spent in N2 sleep
      - `N3 (%)`: Percentage of time spent in N3 sleep

    - **Sleep Disruption Metrics**:
      - `WASO (min)`: Wake After Sleep Onset in minutes
      - `SL (min)`: Sleep Latency in minutes
      - `SFI`: Sleep Fragmentation Index

    - **Arousal Metrics**:
      - `Arousal I.`: Arousal Index

    - **Limb Movement Index (LMI)**:
      - `LMI`: Limb Movement Index

    - **Respiratory Disturbance Metrics**:
      - `AHI`: Overall Apnea-Hypopnea Index 
      - `AHI NREM`: AHI during NREM sleep 
      - `AHI REM`: AHI during REM sleep 
      - `RDI`: Respiratory Disturbance Index
      - `OAI`: Obstructive Apnea Index
      - `CAI`: Central Apnea Index
      - `MAI`: Mixed Apnea Index 
      - `HYI`: Hypopnea Index
      - `RERAI`: Respiratory Effort-Related Arousal Index 

---

### 5. **Cleanup (Optional)**
You can optionally clean up Docker images and containers after running the tasks. This will require reloading/rebuilding the Docker images for future runs.

## Running on GPU

Each module's Conda env ships without bundled CUDA libraries. If you want GPU acceleration, install matched CUDA + cuDNN into the env. The cluster's *driver* is forward-compatible, so you can run TF compiled against an older CUDA toolkit on a newer driver — but TF's bundled cuBLAS only has kernels for the GPU compute capabilities it was built against.

| Env | TF version | Required CUDA / cuDNN | Compatible GPU compute |
| :--- | :--- | :--- | :--- |
| `caisr_stage` | 1.15.0 (Py 3.6) | `cudatoolkit=10.0` + `cudnn=7.6` | sm_60 (Pascal), **sm_70 (Volta — V100)**, sm_75 (Turing — RTX 20-series, T4) |
| `caisr_arousal` | 2.8.0 | `cudatoolkit=11.2` + `cudnn=8.1.0` | sm_60–sm_75 *and* sm_80/86/89 (Ampere, Ada — A100, A30, RTX 30/40-series, RTX 6000 Ada) |

`caisr_resp` and `caisr_limb` are rule-based (scipy/numpy) and **do not benefit from a GPU**.

### One-time CUDA install

Install matching CUDA libs into a *clone* of each env (so you can keep CPU-only envs around for non-GPU runs):

```bash
# Stage — TF 1.15 → only Volta/Turing/Pascal
conda create -n caisr_stage_gpu --clone caisr_stage
conda activate caisr_stage_gpu
conda install -c conda-forge cudatoolkit=10.0 cudnn=7.6
conda deactivate

# Arousal — TF 2.8 → works on all modern GPUs
conda create -n caisr_arousal_gpu --clone caisr_arousal
conda activate caisr_arousal_gpu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda deactivate
```

At runtime, point TF at the env-local CUDA libs:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

To verify a GPU is visible to TF:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If you see `[]`, the env's CUDA libraries don't match your GPU's compute capability — for stage on Ampere/Ada, you'll see `Blas GEMM launch failed` (cuBLAS sm_70 only).

### Stage performance tip

`caisr_stage.py`'s sequential path **builds the TF model once** and reuses it across subjects (saves ~30 s/subject of rebuild + weight reload). On a V100 you should see ~25–50 s/subject steady-state — most of that is CPU-side feature extraction (DE/PSD), not GPU compute.

### Arousal memory stability

`caisr_arousal.py:predict()` calls `tf.keras.backend.clear_session()` at the start of each subject to reset the default TF graph. Without it, model rebuilds accumulate graph nodes across subjects and OOM long-running tasks (~32 GB at ~300 sequential subjects).

## Running at scale with SLURM job arrays

For large cohorts (10k+ recordings), a single sbatch is too slow. Use a job array that slices the input directory across N tasks, each handling subjects whose index-mod-N equals its `SLURM_ARRAY_TASK_ID`. Outputs go to one shared directory; `overwrite=False` makes re-runs idempotent.

### Stage — GPU array (TF 1.15 needs Volta/Turing)

```bash
#!/bin/bash
#SBATCH --job-name=stg_array
#SBATCH --partition=overflow            # Volta is in overflow on this cluster
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda10             # restrict to nodes with the CUDA-10 feature
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-49

set -euo pipefail
TASK_ID=$SLURM_ARRAY_TASK_ID
NTASKS=${SLURM_ARRAY_TASK_COUNT:-50}

REPO=/path/to/CAISR-App
IN=/path/to/cohort/h5
OUT=/path/to/output
SLICE=/scratch/$USER/slices/stg/task_${TASK_ID}

source $(conda info --base)/etc/profile.d/conda.sh
conda activate caisr_stage_gpu
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Build per-task input slice via symlinks
rm -rf "$SLICE" && mkdir -p "$SLICE"
i=0
for h5 in "$IN"/*.h5; do
  if [ $((i % NTASKS)) -eq "$TASK_ID" ]; then
    ln -sfn "$h5" "$SLICE/$(basename "$h5")"
  fi
  i=$((i+1))
done

cd "$REPO"
python caisr_stage.py \
  --input_data_dir "$SLICE" \
  --output_csv_dir "$OUT" \
  --model_dir      "$REPO/stage/models" \
  --param_dir      "$REPO/data/run_parameters"

rm -rf "$SLICE"
```

### Arousal — GPU array (works on Ampere/Ada too)

Same template, but use `caisr_arousal_gpu`, replace `caisr_stage.py` with `caisr_arousal.py`, drop `--model_dir`, and bump `--mem=64G`. If your partition has Ampere/Ada GPUs, drop `--constraint=cuda10`.

### Limb / Resp — CPU sbatch (single node)

Limb and resp are rule-based; submit one CPU sbatch per cohort with `multiprocess=True` (resp uses Ray, limb uses a worker count). No array needed.

### Important: set `multiprocess=False` for GPU array tasks

Each array task gets one GPU. With `multiprocess=True`, the script would spawn many CPU workers, each loading its own TF model into the same GPU — they'd OOM the GPU. Always flip the relevant `data/run_parameters/<task>.csv` to `multiprocess=False` before submitting a GPU array.

### Chaining the full pipeline with dependencies

```bash
# Stage first
STG=$(sbatch --parsable stage_array.sbatch)
# Arousal once stage finishes
ARO=$(sbatch --parsable --dependency=afterany:$STG arousal_array.sbatch)
# A second arousal pass to catch subjects whose stage CSV arrived late
ARO2=$(sbatch --parsable --dependency=afterany:$ARO arousal_array.sbatch)
# Resp once arousal is fully done (resp loads stage_CAISR + arousal_CAISR per subject)
RSP=$(sbatch --parsable --dependency=afterany:$ARO2 resp.sbatch)
# Limb is independent of stage/arousal — fire it in parallel with the chain above
LMB=$(sbatch --parsable limb.sbatch)
# Combine after both resp and limb finish
sbatch --dependency=afterany:$RSP,afterany:$LMB combine.sbatch
```

### Why two arousal passes?

When stage and arousal arrays run in parallel, an arousal task may pass over a subject whose stage CSV hasn't been written yet — the script logs `No stage file found, skipping`. A second arousal pass with `overwrite=False` picks up only those late-stagers and processes them.

## Notes
- **Error Handling**: The script includes basic error handling. If a Docker image cannot be found or loaded, the script will exit with an error message.
- **Modularity**: You can customize the tasks and their order by modifying the `tasks` list in the script.
- **Preprocessing**: If your data is already preprocessed, you can skip the `preprocess` task.

## Troubleshooting
- **Docker Not Found**: Ensure Docker is installed and running. Check by running `docker --version`.
- **Docker Image Not Found**: Make sure all required Docker images are available in the `dockers/` folder or already installed on your system.
- **Apple Silicon (M1/M2/M3/M4)**: The Docker images require x86_64 emulation which is slow and may fail. For native Apple Silicon support without Docker, see [caisr-native](https://github.com/redareda9/caisr-native).

## License
This project is free to use for non-commercial purposes. For commercial use, please contact us directly.

## DeepWiki AI Help for This Repo
Click: [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/bdsp-core/CAISR-App)

![D35F6B1D-5683-4681-A955-B974823B4886](https://github.com/user-attachments/assets/5a9bd25f-f4ce-4df9-bb34-6384766c196a)

## Citation
Please make sure to cite the following paper if you use CAISR in any of your work

> Nasiri, S., Ganglberger, W., Nassi, T., Meulenbrugge, E. J., Moura Junior, V., Ghanta, M., ... & Westover, M. B. (2025).
> CAISR: Achieving Human-Level Performance in Automated Sleep Analysis Across All Clinical Sleep Metrics. Sleep, zsaf134.

## Contact & Support
For support or inquiries, please open an issue on GitHub. If you have any questions or need clarification, the CAISR development team can be contacted via:

- Samaneh Nasiri, PhD
- Wolfgang Ganglberger, PhD
- Thijs-Enagnon Nassi, PhD
- Erik-Jan Meulenbrugge
- Haoqi Sun, PhD
- Robert J Thomas, MD
- M Brandon Westover, MD, PhD

