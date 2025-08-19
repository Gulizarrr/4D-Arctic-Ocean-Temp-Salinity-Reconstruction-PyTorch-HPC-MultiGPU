# 4D Arctic Ocean Temperature & Salinity Reconstruction

This repository contains code and data for a **Neural Network-based 4D reconstruction** of Arctic Ocean temperature and salinity profiles.  
**Note:** The code was developed and run on an HPC environment with multi-GPU support. Scripts may **not run directly** on local machines without modifications.

## Repository Contents

- `last_torch_code.py` : Main PyTorch training and inference script  
- `profiles.pkl` : Sample temperature and salinity profiles  
- `environment.yml` : Conda environment used for HPC  
- `requirements.txt` : Python dependencies  
- `src/` : Additional modules and scripts  
- `Bash_Script_last_torch.txt` : Bash script used for running the training on HPC  
- `README.md` : This file  

## Usage Notes

- The Bash script (`Bash_Script_last_torch.txt`) and multi-GPU training setup are specific to the HPC environment.  
- Running `last_torch_code.py` locally may require modifying device settings (CPU/GPU) and paths.  


## Link to Data

Download the data from [here](https://drive.google.com/drive/folders/1Qfv0EYKHhM5AfTFoUkNyMb4QjftiQdIY?usp=share_link) and place the files in the `data` folder, if needed.

