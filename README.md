# Overview

This repository aims to reproduce the results given in the manuscript "A GPU-accelerated Blocked Adaptive Randomized Range Finder based on an Implicit Householder QR Decomposition" by Carolin Penke and Andreas Herten.

This repository contains
* Stability experiments leading to Fig. 4, in directory `barrf-stability-tests`
    * A script to generate the data `experiment_rangefinder_orthogonality_loss.py`
    * The data in csv format
    * A script to plot the figure from the data  `plot_rangefinder_orthogonality_loss.py`
* Runtime results leading to Fig. 5 - 7
    * Benchmark scripts used to generate these results,
    * Measured runtime results used in the paper in directory `benchmark_results`
    * Plotting scripts to generate figures from results,
* A patch for MAGMA, that fixes matrix generation with given singular values (see [PR](https://github.com/icl-utk-edu/magma/pull/93)),
* This README, describing how to setup the environment.

# Python environment for plots
Python scripts for plotting figures are provided. The uv environment with needed dependencies can be set up by
```
uv sync --locked
```
The plot scripts can be started with
```
uv run --locked plot_xyz.py
```

# Stability experiments
Look at the self-contained directory `reproduce-barrf-householder/barrf-stability-tests` and its README. The stability tests are pure Python code.

# Runtime experiments

## Dependencies
We assume a working CUDA environment, including CuRAND, CuBLAS and nvcc, and a working CPU BLAS implementation. Specifically, we used NVPL. 

Three other repositories are relevant:

1. The main algorithms proposed by the paper (Householder Block Adaptive Randomized Range Finder, HH-BARRF), are found here: https://gitlab.jsc.fz-juelich.de/penke3/householder-block-adaptive-range-finder

2. The legacy code used for comparison, based on Gram-Schmidt-Orthogonalization, comes from [RSVDpack](https://github.com/sergeyvoronin/LowRankMatrixDecompositionCodes). We adapted drivers and routines to have comparable runtime measurements in this [fork](https://github.com/caropen/LowRankMatrixDecompositionCodes).

3. The [MAGMA lbrary](https://github.com/icl-utk-edu/magm) was used in a patched version.

These should be cloned into the `external` directory.

## Reproduce HH-BARRF step by step

Prerquisites: 
NVIDIA CUDA, cuRAND, cuSOLVER, cuBLAS, a CPU BLAS library (we used NVPL).

Before running the benchmark scripts, initialize the environments, e.g. by loading environment modules on an HPC system, and ensure that the installed MAGMA library can be found through `LD_LIBRARY_PATH`. 

### 1. __Clone this repository__

```
git clone https://gitlab.jsc.fz-juelich.de/penke3/reproduce-barrf-householder.git
cd reproduce-barrf-householder
```

### 2. __Install MAGMA library__

From this repo's root:
```
git clone --branch v2.10.0  https://github.com/icl-utk-edu/magma.git external/magma
git -C external/magma apply ../../fix_matrix_generation.patch
``` 

Follow installation guidelines of MAGMA. i.e. create your own make.inc.

```
cd external/magma
make -j 8 lib
make -j 8 testing
make --ignore-errors install [prefix=/usr/local/magma]
```

If you use a non-standard installation directory (prefix), later make sure, this directory is reflected in `$LD_LIBRAY_PATH`. 

### 3. __Clone and compile HH-BARRF__

From this repo's root:
``` 
git clone https://gitlab.jsc.fz-juelich.de/x-dev/householder-block-adaptive-range-finder.git external/householder-block-adaptive-range-finder
cd external/householder-block-adaptive-range-finder
```
The Makefile needs to be adapted to your environment, similar to the Makefiles for MAGMA. You may need to load environment modules or setup OneAPI with `source /opt/intel/oneapi/setvars.sh`. 

```
make -j 8
```

### 4. __Clone and compile forked RSVDpack__

From this repo's root:
```
git clone https://github.com/caropen/LowRankMatrixDecompositionCodes.git external/LowRankMatrixDecompositionCodes
```

The routines used to acquire runtimes are `lapack_code/driver3_justQB.c` for the CPU version and `nvidia_gpu_cublas_code/driver_mkl_and_cublas_justQB_single`. The compiler calls in `oneapi_code/compile_justQB.sh` and `nvidia_gpu_cublas_code/compile_justQB_single.sh` need to be adapted in case icx and MKL are not available.

 ```
 cd external/LowRankMatrixDecompositionCodes/oneapi_code
 make
 cd ../nvidia_gpu_cublas_code/driver_lapack_and_cublas_justQB_single
 make
``` 


### 3. __Benchmarks on HH-BARRF (Fig. 4)__

From this repo's root:
```
bash run_sgeqrr_benchmarks.sh
``` 
The data is saved in `benchmark_results/adaptive_range_finder`. Existing data is overwritten.

Call the plot script to generate a plot as in Fig. 4.

```
uv run --locked plot_sgeqrr_manuscript_heatmaps.py benchmark_results/adaptive_range_finder/
```

### 4. __Benchmarks on RSVDpack (Fig. 5)__
From this repo's root:

```
bash run_randqb_benchmarks.sh
``` 
The data is saved in `benchmark_results/legacy_qb`. Existing data is overwritten. 

Call the plot script to generate a plot as in Fig. 3.

```
uv run --locked plot_randQB_manuscript_heatmaps.py benchmark_results/legacy_qb/
```

### 5. __Benchmarks on fixed-rank range finder (Fig. 6)__ 
From this repo's root:
```
bash run_sgeqrr_nonadaptive_benchmarks.sh
``` 
The data is saved in `benchmark_results/fixed_rank_range_finder`. Existing data is overwritten. 

Call the plot script to generate a plot as in Fig. 3.

```
uv run --locked plot_sgeqrr_nonadaptive_runtimes.py benchmark_results/fixed_rank_range_finder/ --log-y
```
