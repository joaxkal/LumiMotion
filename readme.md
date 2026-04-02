
# LumiMotion - Improving Gaussian Relighting with Scene Dynamics

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)]()
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://joaxkal.github.io/LumiMotion/)
[![Data](https://img.shields.io/badge/Zenodo-Data-blue.svg?logo=zenodo)](https://zenodo.org/records/18894615)  

## Installation

#### 1) Create environment and install CUDA + PyTorch

```bash
git clone --recursive git@github.com:joaxkal/LumiMotion.git
cd LumiMotion

conda create -y --name lumimotion python=3.8.18
conda activate lumimotion

conda install -y cuda-toolkit=12.1 -c nvidia/label/cuda-12.1.0
python -m pip install --upgrade pip
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

#### 2) Install rasterizer and native dependencies

```bash
# Install rasterizer (see notes about versions below!)
pip install git+https://github.com/hbb1/diff-surfel-rasterization.git@f7e2b68a0f17d6d7bca3fa564a96c7678581ce5a

# If rasterizer build fails with glm errors:
# sudo apt install libglm-dev
# # OR
# conda install -y glm -c conda-forge

python -m pip install ./submodules/simple-knn

# install surfel tracer
cd submodules/surfel_tracer && rm -rf ./build && mkdir build && cd build && cmake .. && make && cd ../ && cd ../../

python -m pip install ./submodules/surfel_tracer

python -m pip install -U "setuptools>=64" wheel ninja
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

#### 3) Install Python dependencies

```bash
python -m pip install -r requirements.txt
```
#### ⚠️ Important note on Surfel Rasterizer Versions

Our codebase relies on [this 2DGS surfel rasterizer](https://github.com/hbb1/diff-surfel-rasterization). We did extensive tests and noticed some instabilities in training when switching to newer rasterizer versions. In installation instructions we pinned one commit we found to be stable, to both reproduce the results on synthetic scenes and to support non-pinhole camera for DNA dataset.

However, if you notice problems with training stability or are unable to reproduce our exact results, please follow installation instructions below:

**1. ENERF and Synthetic Datasets**  

All experiments on ENeRF and our synthetic datasets were **initially** run using an older rasterizer version which is included in our repository:   
`pip install submodules/2dgs_rasterizer_lumimotion/diff-surfel-rasterization`  
(this should be the same version as in [this commit](https://github.com/hbb1/diff-surfel-rasterization/commit/661d22484ac4fe15495d34410eab4c635b3ceb74)).

**2. DNA Dataset**  

We later trained LumiMotion on the DNA scenes. Since DNA does **not use a pinhole camera model**, it requires a newer rasterizer that supports such camera types. For these experiments we used:  
`pip install git+https://github.com/hbb1/diff-surfel-rasterization.git@e0ed0207b3e0669960cfad70852200a4a5847f61`.

## Data
Our synthetic dataset is available at: https://zenodo.org/records/18894615. We cannot share ENeRF and DNA data due to signed agreements, but we provide exact instructions how to access and preprocess used scenes. Please refer to `notebooks` folder.

We keep all data in `data` folder:

```
LumiMotion
├── data/
    ├── dna/
    ├── enerf_actors_1_3/
    └── d-nerf-relight-spec32/
```

## Training and evaluation

Stage 1 - Train Geometry For Relighting.   

Stage 2 - Train Albedo, Roughness and Envmap.  

Finally, render materials, nvs and relight, run evaluation.

All configs for synthetic scenes used in paper: `bash_scripts/synthetic_results_from_paper.sh`.    
 
Scripts to run on ENeRF scenes:   
- `bash_scripts/enerf-actor1.sh`,  
- `bash_scripts/enerf-actor3.sh`. 

Scripts to run on DNA scenes:   
- `bash_scripts/dna_shoes.sh`,   
- `bash_scripts/dna_table.sh`,   
- `bash_scripts/dna_hairdryer.sh`. 

If OOM use flag `--load2gpu_on_the_fly`.

## Acknowledgements
We acknowledge the following useful resources and repositories we built upon while developing LumiMotion:
- https://github.com/hustvl/Dynamic-2DGS - codebase for Stage 1
- https://github.com/fudan-zvg/IRGS - codebase for Stage2 (including their surfel tracer)
- https://github.com/NJU-3DV/Relightable3DGaussian - codebase for Stage2.  
- https://github.com/hbb1/diff-surfel-rasterization - Gaussian rasterizer.   

We sincerely thank the authors for making their work open source.  