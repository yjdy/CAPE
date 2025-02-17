# CAPE
source code of paper "A Contextual-Aware Position Encoding for Sequential Recommendation", WWW'25

## Quick Start
1. Run a model on benchmark datasets (e.g., Criteo)

   Users can follow the [benchmark section](#Benchmarking) to get benchmark datasets and running steps for reproducing the existing results. Please see an example here: https://github.com/reczoo/BARS/tree/main/ranking/ctr/DCNv2/DCNv2_criteo_x1

2. Tune hyper-parameters of a model
   
   FuxiCTR currently support fast grid search of hyper-parameters of a model using multiple GPUs. The following example shows the grid search of 8 experiments with 4 GPUs.
    
   ```
   cd experiment
   python run_param_tuner.py --config config/DCN_tiny_parquet_tuner_config.yaml --gpu 0 1 2 3 0 1 2 3
   ```
   
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
