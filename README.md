# PRINT-SAFE: PRINTed ultra-low-cost electronic X-Design with Scalable Adaptive Fault Endurance

This repository contains the code for **"PRINT-SAFE: PRINTed ultra-low-cost electronic X-Design with Scalable Adaptive Fault Endurance,"** published in ESWEEK 2025.

---

## Overview

**PRINT-SAFE** addresses reliability challenges in additive printed electronics (PE) by introducing a novel co-design of training algorithms and hardware for fault-tolerant printed analog neuromorphic circuits (pNCs). Our **Fault-Aware Training (FAT)** method dynamically selects printed activation functions (AFs) to optimize fault endurance and reduce hardware costs.

---

## Key Contributions

- **Co-design of Fault-Tolerant pNCs**: A pioneering approach integrating circuit and algorithmic design for robust printed electronics.

- **Gradient-Based Fault-Aware Training**: Utilizes Gumbel-Softmax for differentiable architecture search, adapting to printing defects.

- **Improved Fault Tolerance**: Achieves significant accuracy improvement (62.1% to 79.4% under 10% fault rate).

- **Resource Optimization**: Reduces power by 54.5%, area by 6.54%, and training time by 56.2% by combining normal and fault-tolerant AFs.

---

## Repository Structure

```plaintext
├── dataset/                # Datasets
├── src/                    # Source code
├── experiment.py           # Main experiment script
├── README.md               # This 
```


---

## SPICE Simulation Data

The **Faulty Behavior Dataset (FBDataset)** is derived from SPICE simulations of non-linear AFs. Ensure this dataset is available for accurate fault modeling.

---

## Running Experiments

Run experiments using `experiment.py` with command-line arguments:

```bash
python3 experiment.py --DATASET 0 --SEED 00 --e_train 0.0 --dropout 0.0 --projectname none_0_FaultAnalysisMixed --type_nonlinear mix --fault_ratio 0.0 --act none
```



## Key Arguments:
- --DATASET: Dataset index.
- --SEED: Random seed.
- --e_train: Training error rate.
- --projectname: Experiment name.
- --type_nonlinear: Type of nonlinearity. Can be normal, robust, or mix.
- --fault_ratio: Fault injection ratio.

## Mix Match Implementation
The mix match strategy is primarily implemented in src/pNN_FA_MIX_MATCH.

## Citation
Please cite our work if you use this code or concepts:

```bibtex
@article{2025PRINTSAFE,
  author    = {Priyanjana Pal and Tara Gheshlaghi and Haibin Zhao and Michael Hefenbrock and Michael Beigl and Mehdi B. Tahoori},
  title     = {{PRINT-SAFE: PRINTed ultra-low-cost electronic X-Design with Scalable Adaptive Fault Endurance}},
  journal   = {ESWEEK 2025},
}
```


