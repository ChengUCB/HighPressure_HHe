# HighPressure_HHe

Supporting information for “Hydrogen–helium immiscibility boundary in planets”.

### Repository layout

```
HighPressure_HHe/
├─ DFT_calculations/        # Training dataset from DFT
├─ MLPs/                    # Training scripts and fitted MLPs (N2P2, CACE, MACE)
├─ MLPs_benchmark/          # Benchmarking the fitted MLPs
├─ MLP_MD/                  # MD inputs, runs, and analysis scripts
└─ MLP-MD_results/          # Immiscibility boundaries and Redlich–Kister fits
```

## Training dataset 
The training dataset is provided in the ./DFT_calculations directory.

## MLP packages 
- N2P2: [https://github.com/CompPhysVienna/n2p2](https://github.com/CompPhysVienna/n2p2)

- CACE: [https://github.com/BingqingCheng/cace](https://github.com/BingqingCheng/cace)

- MACE: [https://github.com/ACEsuit/mace](https://github.com/ACEsuit/mace)
  
Training scripts and fitted MLPs are provided in ./MLPs

## MLP MD and related analysis scripts
Molecular dynamics inputs, runs, and analysis utilities are provided in the ./MLP_MD directory.

As for the S0 method, please also refer to: https://github.com/BingqingCheng/S0

## Results 
The ./MLP-MD_results directory contains determined immiscibility boundaries and Redlich–Kister model fits.

