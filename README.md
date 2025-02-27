# CAPE
source code of paper "A Contextual-Aware Position Encoding for Sequential Recommendation", WWW'25

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2409.12740-da282a.svg)](https://arxiv.org/pdf/2502.09027)
<a href="https://github.com/reczoo/FuxiCTR/blob/main/LICENSE"><img src="https://img.shields.io/github/license/reczoo/fuxictr.svg" style="max-width: 100%;" alt="License"></a>
<a href="https://pypi.org/project/fuxictr"><img src="https://img.shields.io/badge/python-3.9+-blue" style="max-width: 100%;" alt="Python version"></a>
<a href="https://pypi.org/project/fuxictr"><img src="https://img.shields.io/badge/pytorch-1.10+-blue" style="max-width: 100%;" alt="Pytorch version"></a>
</div>

<p align="center"> 
    <img src="./figs/cape3.PNG" width="800">
</p>

**The code in this repository is currently under development**

## Quick Start
1. Run a model on benchmark datasets (e.g., KuaiVideo)

   Users can follow the [benchmark section](#Benchmarking) to get benchmark datasets and running steps for reproducing the existing results. Please see an example here: https://github.com/reczoo/BARS/tree/main/ranking/ctr/DCNv2/DCNv2_criteo_x1

2. Tune hyper-parameters of a model
    
   ```
   cd experiment
   python run_param_tuner.py --config config/DIN_amazonelectronics_x1_tuner_config.yaml --gpu 0 
   ```
   You can set hyper-parameter *use_cope* to True to use CAPE in the model.

## Experimental Result

| SR Model | PE     | AmazonElectronics (gAUC↑) | AmazonElectronics (AUC↑) | AmazonElectronics (logloss↓) | KuaiVideo (gAUC↑) | KuaiVideo (AUC↑) | KuaiVideo (logloss↓) |
|----------|--------|--------------------------|--------------------------|-----------------------------|--------------------|------------------|----------------------|
| **DIN**  | None   | 0.883526                 | 0.886028                 | 0.43019                    | 0.661646           | 0.741604         | 0.447621             |
|          | Naïve  | 0.883947                 | 0.886702                 | <u>0.428511</u>            | 0.661123           | 0.743029         | 0.441684             |
|          | RoPE   | 0.884544                 | <u>0.887220</u>          | 0.429256                   | <u>0.664594</u>    | <u>0.745957</u>  | <u>0.439012</u>      |
|          | CoPE   | <u>0.884549</u>          | 0.886706                 | 0.430099                   | 0.661646           | 0.742936         | 0.442531             |
|          | CAPE   | **0.885698**             | **0.888156**             | **0.428468**               | **0.665215**       | **0.745973**     | **0.438510**         |
| **DIEN** | None   | 0.884014                 | 0.886774                 | 0.428798                   | 0.661032           | 0.743491         | 0.437697             |
|          | Naïve  | 0.885397                 | 0.887903                 | <u>0.426713</u>            | 0.659564           | <u>0.744395</u>  | <u>0.435241</u>      |
|          | RoPE   | <u>0.887128</u>          | <u>0.889513</u>          | 0.426858                   | 0.661536           | 0.744089         | 0.438067             |
|          | CoPE   | 0.885736                 | 0.888723                 | 0.426941                   | <u>0.661589</u>    | 0.744392         | 0.435911             |
|          | CAPE   | **0.887736**             | **0.889723**             | **0.425941**               | **0.662178**       | **0.744486**     | **0.434926**         |
| **BST**  | None   | 0.878645                 | 0.879191                 | 0.464533                   | 0.661409           | 0.741465         | 0.446131             |
|          | Naïve  | 0.881701                 | 0.884188                 | 0.430641                   | 0.660665           | 0.744091         | 0.435769             |
|          | RoPE   | <u>0.882918</u>          | <u>0.885657</u>          | **0.429100**               | <u>0.662909</u>    | <u>0.745502</u>  | **0.432553**         |
|          | CoPE   | 0.882166                 | 0.884399                 | <u>0.430178</u>            | 0.660777           | 0.744202         | 0.435414             |
|          | CAPE   | **0.883349**             | **0.886499**             | 0.430582                   | **0.664139**       | **0.746326**     | <u>0.433429</u>      |
| **SASRec**| None  | 0.879654                 | 0.879692                 | 0.460631                   | 0.659623           | 0.744130         | 0.437802             |
|          | Naïve  | 0.879493                 | 0.882471                 | 0.434959                   | 0.661535           | 0.743983         | 0.436454             |
|          | RoPE   | <u>0.882185</u>          | <u>0.884663</u>          | 0.433952                   | <u>0.661662</u>    | <u>0.745082</u>  | <u>0.434806</u>      |
|          | CoPE   | 0.880922                 | 0.883757                 | <u>0.433889</u>            | 0.654668           | 0.741277         | **0.433345**         |
|          | CAPE   | **0.882793**             | **0.885344**             | **0.431368**               | **0.663006**       | **0.745512**     | 0.435691             |
| **DMIN** | None   | 0.885558                 | 0.887029                 | 0.427322                   | 0.659194           | 0.743807         | 0.435644             |
|          | Naïve  | 0.883131                 | 0.885329                 | 0.431241                   | <u>0.661405</u>    | <u>0.745304</u>  | **0.434112**         |
|          | RoPE   | 0.884383                 | 0.886762                 | <u>0.425995</u>            | 0.660126           | 0.745036         | 0.433466             |
|          | CoPE   | <u>0.885583</u>          | <u>0.887662</u>          | 0.426295                   | 0.659226           | 0.744025         | 0.434486             |
|          | CAPE   | **0.885703**             | **0.888053**             | **0.424144**               | **0.662567**       | **0.746088**     | <u>0.434272</u>      |

## Citation

If our work has been of assistance to your work, feel free to give us a star ⭐ or cite us using :  

```
@article{yuan2025CAPE,
      title={A Contextual-Aware Position Encoding for Sequential Recommendation}, 
      author={Jun Yuan and Guohao Cai and Zhenhua Dong},
      journal={arXiv preprint arXiv:2502.09027},
      year={2025},
      eprint={2502.09027},
      archivePrefix={arXiv}
}
```

## Aknowledgement
> Thanks to the excellent code repository [FuxiCTR](https://github.com/reczoo/FuxiCTR) 
> CAPE is released under the Apache-2.0 license, some codes are modified from FuxiCTR, which are released under the Apache-2.0 license.
