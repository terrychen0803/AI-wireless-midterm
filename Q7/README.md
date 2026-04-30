# Exercise 2.15 — CsiNet Generalization across Channel Datasets

The work is to see**How CsiNet performs CSI reconstruction generalization ability on different channel dataset**。And finished the following questions

(a) Please adopt the COST 2100 channel model to generate more than five different channel datasets referring
to [14], such as changing the distribution of users.

(b) Evaluate the CSI reconstruction NMSE of the trained CsiNet model on each of these datasets.

(c) Please mix the above different channel datasets and use them to train CsiNet. Then, compare the recon-
struction performance with that in (b). Based on the results, consider how to improve the generalization of
the feedback methods for complex channels in practical systems.


---

## 1. Project Structure

```text
exercise_2_15_solution/
│
├── generate_cost2100_like_datasets.py     # Generate multiple channel datasets
├── csinet_experiment.py                   # Train/test CsiNet and plot results
├── requirements.txt                       # Python package requirements
├── README.md                              # This document
│
├── data/                                  # Generated datasets
│   ├── cell_uniform/
│   │   ├── train.mat
│   │   ├── val.mat
│   │   └── test.mat
│   ├── center_uniform/
│   ├── edge_uniform/
│   ├── left_half/
│   ├── right_half/
│   ├── two_hotspots/
│   └── diagonal_corridor/
│
└── results/                               # Training/testing outputs
    ├── baseline_cell_uniform_dim128.keras
    ├── mixed_all_dim128.keras
    ├── baseline_train_cell_uniform_dim128.csv
    ├── mixed_train_all_dim128.csv
    ├── log_baseline_cell_uniform_dim128.csv
    ├── log_mixed_all_dim128.csv
    └── compare_dim128.png
```

---

## 2. Environment Setup

The program was designed for Python and TensorFlow/Keras.

Recommended environment:

```text
Python 3.10 / 3.11 / 3.12
TensorFlow
NumPy
SciPy
Matplotlib
Pandas
```

### Windows PowerShell setup

```powershell
cd C:\python_projects
Expand-Archive -Path .\exercise_2_15_solution.zip -DestinationPath .\exercise_2_15_solution -Force
cd .\exercise_2_15_solution

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Verify TensorFlow:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 3. Program Explanation

### 3.1 `generate_cost2100_like_datasets.py`

This file generates multiple channel datasets by changing the spatial distribution of users. The generated CSI data follows a clustered multipath channel concept similar to COST 2100. Each channel sample is transformed into the angular-delay domain and stored in the same format used by the CsiNet example.

The generated channel matrix size is:

```text
32 × 32 × 2
```

where:

```text
32 × 32 : angular-delay CSI matrix
2       : real part and imaginary part
```

After flattening, each CSI sample becomes:

```text
32 × 32 × 2 = 2048
```

Therefore, each `.mat` file stores the variable:

```text
HT.shape = [number_of_samples, 2048]
```

The program generates seven datasets:

| Dataset | User distribution description |
|---|---|
| `cell_uniform` | Users are uniformly distributed in the whole cell. |
| `center_uniform` | Users are concentrated near the BS. |
| `edge_uniform` | Users are concentrated near the cell edge. |
| `left_half` | Users are distributed only in the left half of the cell. |
| `right_half` | Users are distributed only in the right half of the cell. |
| `two_hotspots` | Users are concentrated around two hotspot regions. |
| `diagonal_corridor` | Users are distributed along a diagonal corridor. |

This satisfies requirement **(a)** because more than five different channel datasets are generated.

---

### 3.2 `csinet_experiment.py`

This file includes the CsiNet model, training procedure, testing procedure, NMSE calculation, and plotting function.

The CsiNet architecture used in this experiment is:

```text
Input CSI: 32 × 32 × 2
↓
Conv2D feature extraction
↓
Flatten
↓
Dense layer encoder
↓
Compressed codeword
↓
Dense layer decoder
↓
Reshape to 32 × 32 × 2
↓
RefineNet residual blocks
↓
Final Conv2D layer
↓
Reconstructed CSI: 32 × 32 × 2
```

The model is trained as an autoencoder:

```python
model.fit(x_train, x_train)
```

This means the input and output target are both the original CSI matrix. The neural network learns:

```text
CSI matrix → compressed codeword → reconstructed CSI matrix
```

The compression dimension used in this experiment is:

```text
encoded_dim = 128
```

Since the original input dimension is 2048, the compression ratio is:

```text
128 / 2048 = 1 / 16
```

---

## 4. How to Run the Program

### Step 1 — Generate channel datasets

For a small quick test:

```powershell
python generate_cost2100_like_datasets.py --n-train 1000 --n-val 200 --n-test 200
```

For a more formal experiment:

```powershell
python generate_cost2100_like_datasets.py --n-train 10000 --n-val 2000 --n-test 2000
```

This creates:

```text
data/<dataset_name>/train.mat
data/<dataset_name>/val.mat
data/<dataset_name>/test.mat
```

---

### Step 2 — Train baseline CsiNet and test on all datasets

```powershell
python csinet_experiment.py --mode baseline --train-dataset cell_uniform --encoded-dim 128 --epochs 100
```

This command trains CsiNet only on the `cell_uniform` dataset, then evaluates it on all seven datasets.

Output files:

```text
results/baseline_cell_uniform_dim128.keras
results/baseline_train_cell_uniform_dim128.csv
results/log_baseline_cell_uniform_dim128.csv
```

This corresponds to requirement **(b)**.

---

### Step 3 — Train mixed CsiNet and test on all datasets

```powershell
python csinet_experiment.py --mode mixed --encoded-dim 128 --epochs 100
```

This command mixes all seven datasets for training and then evaluates the trained model on all seven test datasets.

Output files:

```text
results/mixed_all_dim128.keras
results/mixed_train_all_dim128.csv
results/log_mixed_all_dim128.csv
```

This corresponds to requirement **(c)**.

---

### Step 4 — Plot comparison figure

```powershell
python csinet_experiment.py --mode plot --encoded-dim 128
```

Output file:

```text
results/compare_dim128.png
```

---

## 5. Evaluation Metric

The evaluation metric is normalized mean square error, NMSE:

```text
NMSE = ||H - H_hat||² / ||H||²
```

In the result table, NMSE is converted to dB:

```text
NMSE_dB = 10 log10(NMSE)
```

A lower NMSE value means better CSI reconstruction. Since the values are in dB, a more negative value is better.

For example:

```text
-21 dB is better than -15 dB
```

---

## 6. Results

Experiment setting:

```text
Compression dimension: encoded_dim = 128
Compression ratio: 1/16
Training epochs: 100
Baseline training dataset: cell_uniform
Mixed training dataset: all seven datasets
```

---

### 6.1 Requirement (a): Generated Datasets

The following seven datasets were generated by changing the UE/user distribution:

| No. | Dataset | Description |
|---:|---|---|
| 1 | `cell_uniform` | Uniform distribution over the whole cell. |
| 2 | `center_uniform` | Users concentrated near the BS. |
| 3 | `edge_uniform` | Users concentrated near the cell edge. |
| 4 | `left_half` | Users located in the left half region. |
| 5 | `right_half` | Users located in the right half region. |
| 6 | `two_hotspots` | Users concentrated around two hotspot areas. |
| 7 | `diagonal_corridor` | Users distributed along a diagonal corridor. |

Therefore, the program satisfies the requirement of generating more than five different channel datasets.

---

### 6.2 Requirement (b): Baseline CsiNet Tested on Different Datasets

The baseline model was trained only on the `cell_uniform` dataset. It was then tested on all seven datasets.

| Test dataset | Baseline NMSE (dB) |
|---|---:|
| `cell_uniform` | -15.9553 |
| `center_uniform` | -12.2262 |
| `edge_uniform` | -16.5886 |
| `left_half` | -15.8365 |
| `right_half` | -15.9078 |
| `two_hotspots` | -14.0999 |
| `diagonal_corridor` | -11.5649 |
| **Average** | **-14.5970** |

#### Analysis of baseline result

The baseline CsiNet model performs reasonably well on datasets whose channel distribution is similar to the training distribution. However, its performance becomes worse when the testing user distribution is very different from the training dataset.

The worst result occurs on:

```text
diagonal_corridor: -11.5649 dB
```

This is expected because the diagonal-corridor distribution has a more constrained and non-uniform user geometry. The model trained only on uniform users has not seen enough channel samples from this kind of spatial distribution, so its reconstruction ability degrades.

The `center_uniform` dataset also has weaker performance:

```text
center_uniform: -12.2262 dB
```

This shows that even though the channel generator uses the same general model, changing the user location distribution can still cause a distribution shift. CsiNet is a data-driven method, so its performance strongly depends on whether the training data distribution covers the testing channel distribution.

---

### 6.3 Requirement (c): Mixed Dataset Training Result

The mixed model was trained using all seven datasets together. It was then tested on the same seven datasets.

| Test dataset | Baseline NMSE (dB) | Mixed-training NMSE (dB) | Improvement (dB) |
|---|---:|---:|---:|
| `cell_uniform` | -15.9553 | -21.3079 | 5.3526 |
| `center_uniform` | -12.2262 | -19.2815 | 7.0553 |
| `edge_uniform` | -16.5886 | -21.5712 | 4.9827 |
| `left_half` | -15.8365 | -21.2104 | 5.3739 |
| `right_half` | -15.9078 | -21.2571 | 5.3493 |
| `two_hotspots` | -14.0999 | -19.8751 | 5.7753 |
| `diagonal_corridor` | -11.5649 | -17.3553 | 5.7904 |
| **Average** | **-14.5970** | **-20.2655** | **5.6685** |

#### Analysis of mixed-training result

The mixed-trained CsiNet model achieves better NMSE on every dataset. The average NMSE improves from:

```text
Baseline average: -14.5970 dB
Mixed average:    -20.2655 dB
```

The average improvement is approximately:

```text
5.6685 dB
```

This shows that using diverse training data improves the generalization capability of CsiNet. The baseline model only learns CSI structures from one user distribution, while the mixed model learns channel characteristics from multiple spatial distributions.

The largest improvement appears on:

```text
center_uniform: 7.0553 dB improvement
```

This means that the baseline model has difficulty reconstructing channels from center-concentrated users, but the mixed training set provides enough examples for the model to learn this distribution.

The `diagonal_corridor` dataset remains the most difficult case after mixed training:

```text
diagonal_corridor mixed NMSE: -17.3553 dB
```

However, it still improves by about:

```text
5.7904 dB
```

This confirms that mixed-dataset training does not completely eliminate distribution mismatch, but it significantly reduces its impact.

---

## 7. Training Log Summary

The training logs also show that the mixed model converges to a lower validation loss.

| Model | Final train loss | Final validation loss | Best validation loss | Best epoch |
|---|---:|---:|---:|---:|
| Baseline CsiNet | 3.3432e-05 | 5.2801e-05 | 4.4043e-05 | 93 |
| Mixed CsiNet | 1.3607e-05 | 1.6078e-05 | 1.5293e-05 | 96 |

The mixed model has both lower final training loss and lower validation loss. This means the model trained on diverse channel datasets learns a more general CSI reconstruction function.

---

## 8. Figure Analysis

The comparison figure is saved as:

```text
results/compare_dim128.png
```

The figure compares baseline and mixed-training NMSE across all datasets.

![Cimage]([results/compare_dim128.png](https://github.com/terrychen0803/AI-wireless-midterm/blob/main/Q7/result/compare_dim128.png))

In the figure:

- Blue bars represent the baseline model trained only on `cell_uniform`.
- Orange bars represent the model trained using mixed datasets.
- The y-axis is NMSE in dB.
- Lower bars indicate better reconstruction performance.

The orange bars are consistently lower than the blue bars for all datasets. Therefore, mixed training provides better reconstruction accuracy and better robustness across different channel distributions.

---

## 9. Discussion: How to Improve Generalization in Practical CSI Feedback Systems

From the results, CsiNet is sensitive to the distribution of training data. If the model is trained only under one fixed user distribution, its performance may degrade when the practical channel distribution changes.

To improve generalization in practical systems, several methods can be considered:

1. **Diverse channel training data**  
   The training dataset should include different user distributions, distances, angles, scattering patterns, and propagation environments.

2. **Mixed-scenario training**  
   Instead of training CsiNet on only one channel scenario, multiple indoor/outdoor, near/far, center/edge, and hotspot distributions should be mixed during training.

3. **Domain adaptation or fine-tuning**  
   If the deployment environment is different from the original training environment, the model can be fine-tuned using a small number of measured CSI samples from the new environment.

4. **Data augmentation**  
   Random perturbation of path gains, delays, AoAs, and noise levels can help the neural network learn more robust channel features.

5. **Environment-aware model design**  
   The system may use different CsiNet models for different scenarios, or condition the model on environment information such as user location region or mobility state.

6. **Online update**  
   In practical wireless systems, the channel distribution may slowly change over time. Periodic retraining or online adaptation can help maintain CSI reconstruction performance.

---

## 10. Conclusion

This experiment shows that CsiNet can reconstruct compressed CSI effectively, but its performance depends strongly on the training channel distribution. When the model is trained only on a single user distribution, the reconstruction NMSE becomes worse under distribution shifts. After mixing different channel datasets for training, the NMSE improves across all testing datasets.

The average NMSE improves from:

```text
-14.5970 dB to -20.2655 dB
```

with an average improvement of:

```text
5.6685 dB
```

Therefore, for practical CSI feedback systems, the training data should cover diverse channel conditions. Mixed-scenario training is a simple and effective method to improve the generalization ability of CsiNet.

---

## 11. Main Files Generated in This Experiment

| File | Description |
|---|---|
| `baseline_cell_uniform_dim128.keras` | CsiNet model trained only on `cell_uniform`. |
| `mixed_all_dim128.keras` | CsiNet model trained on all mixed datasets. |
| `baseline_train_cell_uniform_dim128.csv` | Baseline testing NMSE on all datasets. |
| `mixed_train_all_dim128.csv` | Mixed-training testing NMSE on all datasets. |
| `log_baseline_cell_uniform_dim128.csv` | Training log of baseline CsiNet. |
| `log_mixed_all_dim128.csv` | Training log of mixed CsiNet. |
| `compare_dim128.png` | NMSE comparison figure. |

---

## 12. Short English Summary for Report

In this experiment, seven different channel datasets were generated by changing the user distribution under a COST2100-like clustered channel model. A baseline CsiNet model was first trained only on the cell-uniform dataset and evaluated on all datasets. The results show that the baseline model suffers from distribution mismatch, especially for center-concentrated and diagonal-corridor user distributions.

After mixing all datasets and retraining CsiNet, the reconstruction NMSE improves on every test dataset. The average NMSE improves from -14.5970 dB to -20.2655 dB, corresponding to an average gain of about 5.6685 dB. These results indicate that mixed-scenario training improves the generalization capability of CsiNet for practical CSI feedback systems.

