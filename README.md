# CAISR GitHub Readme

## Introduction
The **CAISR system** is designed to streamline tasks related to sleep data analysis by using Docker containers for ease of use. The primary method of using this system is by downloading pre-built Docker images from our website. However, the Python code used to create these images is also available here for transparency and customization purposes. Users who wish to adapt the system to their own datasets or analysis preferences can rebuild the Docker images with minimal effort.

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
    `f3-m2`, `f2-m1`, `c3-m2`, `c4-m1`, `o1-m2`, `o2-m1`, or `eeg`.
  - CAISR performs best when six EEG channels are provided:  
    `f3-m2`, `f2-m1`, `c3-m2`, `c4-m1`, `o1-m2`, `o2-m1`.

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

## Notes
- **Error Handling**: The script includes basic error handling. If a Docker image cannot be found or loaded, the script will exit with an error message.
- **Modularity**: You can customize the tasks and their order by modifying the `tasks` list in the script.
- **Preprocessing**: If your data is already preprocessed, you can skip the `preprocess` task.

## Troubleshooting
- **Docker Not Found**: Ensure Docker is installed and running. Check by running `docker --version`.
- **Docker Image Not Found**: Make sure all required Docker images are available in the `dockers/` folder or already installed on your system.

## License
This project is free to use for non-commercial purposes. For commercial use, please contact us directly.

## DeepWiki AI Help for This Repo
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/bdsp-core/CAISR-App)


## Contact & Support
For support or inquiries, please open an issue on GitHub. If you have any questions or need clarification, the CAISR development team can be contacted via:

- Samaneh Nasiri, PhD
- Wolfgang Ganglberger, PhD
- Thijs-Enagnon Nassi, PhD
- Erik-Jan Meulenbrugge
- Haoqi Sun, PhD
- Robert J Thomas, MD
- M Brandon Westover, MD, PhD

