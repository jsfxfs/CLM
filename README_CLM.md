<p align="center">
    <!-- license badge -->
    <a href="https://github.com/nerfstudio-project/nerfstudio/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
    <!-- stars badge -->
    <a href="https://github.com/nyu-systems/Grendel-GS/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/nyu-systems/Grendel-GS?style=social"/>
    </a>
    <!-- community badges -->
<!--     <a href="https://discord.gg/uMbNqcraFc"><img src="https://dcbadge.vercel.app/api/server/uMbNqcraFc?style=plastic"/></a> -->
    <!-- last commit badge -->
<!--     <a href="https://github.com/TarzanZhao/Dist-GS/commits/master">
        <img alt="Last Commit" src="https://img.shields.io/github/last-commit/TarzanZhao/Dist-GS"/>
    </a> -->
    <!-- pull requests badge -->
    <a href="https://github.com/nyu-systems/Grendel-GS/pulls">
        <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/nyu-systems/Grendel-GS"/>
    </a>

</p>


<div align="center">

Grendel-GS
===========================
_<h4>Gaussian Splatting at Scale with Distributed Training System</h4>_

### [Paper](https://arxiv.org/abs/2406.18533) | [Project Page](https://daohanlu.github.io/scaling-up-3dgs/)

<div align="left">

<div align="center">
    <img src="assets/teaser.png" width="900">
</div>

<details>
  <summary> <strong> Click Here to Download Pre-trained Models behind the above visualizations </strong> </summary>
  
  - [Pre-trained Rubble Model (On the left)](https://3dgs-public.s3.amazonaws.com/Grendal-GS-checkpoints-and-evalutation-images/psnr273-rubble-pointcloud.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAXBPATLFWQSADNSG5%2F20240707%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240707T053805Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=74f9dc5c0346a1b68c05b8bb7da0dbe430a7cdda1c9732f2e17e3c72ab1e13ba), [Corresponding Evaluation Images](https://3dgs-public.s3.amazonaws.com/Grendal-GS-checkpoints-and-evalutation-images/psnr273-rubble-image.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAXBPATLFWQSADNSG5%2F20240707%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240707T053756Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=751f3236b3657acc09e7ed6bddb6e5493103fc4184165d2304abc9821535a5d3)
  - [Pre-trained MatrixCity Model (On the right)](https://3dgs-public.s3.amazonaws.com/Grendal-GS-checkpoints-and-evalutation-images/psnr270-matrixcity-blockall-pointcloud.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAXBPATLFWQSADNSG5%2F20240707%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240707T053813Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=784b1211e89a4c525a44ebf5bfd53d0ceb0370fc2aa9bf32546545e79b8f23d3), [Corresponding Evaluation Images](https://3dgs-public.s3.amazonaws.com/Grendal-GS-checkpoints-and-evalutation-images/psnr270-matrixcity-blockall-images.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAXBPATLFWQSADNSG5%2F20240707%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240707T053809Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=72598685d26b557bca4df653053368d1fc15b9b8cb1d094e0fad89460a5b8d8d)

</details>

<!-- 
### [Rubble Model ](https://www.wpeebles.com/DiT) | [Rubble Evaluation Images](https://www.wpeebles.com/DiT) | [MatrixCity Model](https://www.wpeebles.com/DiT) |  [MatrixCity Evaluation Images](https://www.wpeebles.com/DiT) 

### [Pre-trained Rubble Model(the left one)](https://www.wpeebles.com/DiT) | [Rubble Evaluation Images(the left one)](https://www.wpeebles.com/DiT) | [Pre-trained MatrixCity Model(the right one)](https://www.wpeebles.com/DiT) |  [MatrixCity Evaluation Images](https://www.wpeebles.com/DiT) 
-->



# Overview

We design and implement **Grendel-GS**, which serves as a distributed implementation of 3D Gaussian Splatting training. We aim to help 3DGS achieve its *scaling laws* with distributed system support, just as the achievements of current LLMs rely on distributed system. 

By using Grendel, your 3DGS training could leverage multiple GPUs' capability to achieve significantly ***faster training***, supports a substantially ***more Gaussians*** in GPU memory, and ultimately allows for the reconstruction of ***larger-area***, ***higher-resolution*** scenes to better PSNR. Grendel-GS retains the original algorithm, making it a ***direct and safe replacement*** for original 3DGS implementation in any Gaussian Splatting workflow or application.

For examples, with 4 GPU, Grendel-GS allows you to:
- Train Mip360 >3.5 times faster.
- Support directly training large-scale 4K scenes(Mega-NeRF Rubble) using >40 millions gaussians without OOM.
- Train the Temple\&Tanks Truck scene to PSNR 23.79 within merely 45 seconds (on 7000 images)


<!-- 
*Many more new features are developing now, following us!*
-->
> ### 📢 News 
> - 7.15.2024 - We now support gsplat as the CUDA backend during training!

> ### 🌟 Follow us for future updates! Interested in collaborating or contributing? [**Email us!**](mailto:hz3496@nyu.edu)

**Table of contents**
-----
- [Why use Grendel-GS](#why-use-Grendel-gs)
- [How to use Grendel-GS](#how-to-use-Grendel-gs)
    - [Setup](#setup)
    - [Training](#training)
    - [Render Pretrained-Model](#rendering)
    - [Calculate Metrics](#evaluating-metrics)
    - [Migrating from original 3DGS codebase](#migrating-from-original-3dgs-codebase)
- [Benefits and Examples](#benefits-and-examples)
- [Paper](#paper-and-citation)
- [License](#license)
- [Reference](#reference)
------

# Why use Grendel-GS

Here is a diagram showing why you may need distributed gaussian splatting training like our Grendel-GS' techniques:


![whydistributed](https://github.com/nyu-systems/Grendel-GS/assets/45677459/4237777e-662c-4611-9196-f60fcae927d6)



# How to use Grendel-GS

This repo and its dependency, our customized distributed version of rendering cuda code([diff-gaussian-rasterization](https://github.com/nyu-systems/diff-gaussian-rasterization)), are both forks from the [original 3DGS implementation](https://github.com/graphdeco-inria/gaussian-splatting). Therefore, the usage is generally very similar to the original 3DGS. 

The two main differences are:

1. We support training on multiple GPUs, using the `torchrun` command-line utility provided by PyTorch to launch jobs.
2. We support batch sizes greater than 1, with the `--bsz` argument flag used to specify the batch size.


<!-- 
Our repository contains a distributed PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs. The optimizer uses PyTorch and CUDA extensions to produce trained models. 
-->

## Setup

### Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
git clone git@github.com:nyu-systems/Grendel-GS.git --recursive
```

### Pytorch Environment

Ensure you have Conda, GPU with compatible driver and cuda environment installed on your machine, as prerequisites. Then please install `PyTorch`, `Torchvision`, `Plyfile`, `tqdm` which are essential packages. Make sure PyTorch version >= 1.10 to have torchrun for distributed training. Finally, compile and install two dependent cuda repo `diff-gaussian-rasterization` and `simple-knn` containing our customized cuda kernels for rendering and etc.

We provide a yml file for easy environment setup. However, you should choose the versions to match your local running environment. 
```
conda env create --file environment.yml
conda activate gaussian_splatting
```

NOTES: We kept additional dependencies minimal compared to the original 3DGS. For environment setup issues, maybe you could refer to the [original 3DGS repo issue section](https://github.com/graphdeco-inria/gaussian-splatting/issues) first.

## Migrating from original 3DGS codebase

If you are currently using the original 3DGS codebase for training in your application, you can effortlessly switch to our codebase because we haven't made any algorithmic changes. This will allow you to train faster and successfully train larger, higher-precision scenes without running out of memory (OOM) within a reasonable time frame. 

It is worth noting that we only support the training functionality; this repository does not include the interactive viewer, network viewer, or colmap features from the original 3DGS. We are actively developing to support more features. Please let us know your needs or directly contribute to our project. Thank you!

---



# Benefits and Examples

## Significantly Faster Training Without Compromising Reconstruction Quality On Mip360 Dataset

### Training Time

| 30k Train Time(min)   |   stump |   bicycle |   kitchen |   room |   counter |   garden |   bonsai |
|:----------------------|--------:|----------:|----------:|-------:|----------:|---------:|---------:|
| 1 GPU + Batch Size=1  |   24.03 |     30.18 |     25.58 |  22.45 |     21.6  |    30.15 |    19.18 |
| 4 GPU + Batch Size=1  |    9.07 |     11.67 |      9.53 |   8.93 |      8.82 |    10.85 |     8.03 |
| 4 GPU + Batch Size=4  |    5.22 |      6.47 |      6.98 |   6.18 |      5.98 |     6.48 |     5.28 |

### Test PSNR

| 30k Test PSNR        |   stump |   bicycle |   kitchen |   room |   counter |   garden |   bonsai |
|:---------------------|--------:|----------:|----------:|-------:|----------:|---------:|---------:|
| 1 GPU + Batch Size=1 |   26.61 |     25.21 |     31.4  |  31.4  |     28.93 |    27.27 |    32.01 |
| 4 GPU + Batch Size=1 |   26.65 |     25.19 |     31.41 |  31.38 |     28.98 |    27.28 |    31.92 |
| 4 GPU + Batch Size=4 |   26.59 |     25.17 |     31.37 |  31.32 |     28.98 |    27.2  |    31.94 |
---

### Reproduction Instructions

1. Download and unzip the [Mip360 dataset](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip).
2. Activate the appropriate conda/python environment.
3. To execute all experiments and generate this table, run the following command:
   ```bash
   bash examples/mip360/eval_all_mip360.sh <path_to_save_experiment_results> <path_to_mip360_dataset>
   ```

## Significantly Speed up and Reduce per-GPU memory usage on Mip360 at *4K Resolution*

| Configuration                  | 50k Training Time   |   Memory Per GPU |   PSNR |
|:-------------------------------|:--------------------|-----------------:|-------:|
| bicycle + 1 GPU + Batch Size=1 | 2h 38min            |            37.18 |  23.78 |
| bicycle + 4 GPU + Batch Size=1 | 0h 50min            |            10.39 |  23.79 |
| garden + 1 GPU + Batch Size=1  | 2h 49min            |            29.87 |  26.06 |
| garden + 4 GPU + Batch Size=1  | 0h 50min            |             7.88 |  26.06 |

Unlike the typical approach of downsampling the Mip360 dataset by a factor of four before training, our system can train directly at full resolution. The bicycle and garden images have resolutions of 4946x3286 and 5187x3361, respectively. Our distributed system demonstrates that we can significantly accelerate and reduce memory usage per GPU by several folds without sacrificing quality.

### Reproduction Instructions

Set up the dataset and Python environment as outlined previously, then execute the following:
```bash
   bash examples/mip360_4k/eval_mip360_4k.sh <path_to_save_experiment_results> <path_to_mip360_dataset>
   ```

## Train in 45 Seconds on Tanks&Temple at *1K Resolution*

| Configuration                | 7k Training Time   |   7k test PSNR | 30k Training Time   |   30k test PSNR |
|:-----------------------------|:-------------------|---------------:|:--------------------|----------------:|
| train + 4 GPU + Batch Size=8 | 44s                |          19.37 | 3min 30s            |           21.87 |
| truck + 4 GPU + Batch Size=8 | 45s                |          23.79 | 3min 39s            |           25.35 |

Tanks&Temples dataset includes train and truck scenes with resolutions of 980x545 and 979x546, respectively. Utilizing 4 GPUs, we've managed to train on these small scenes to a reasonable quality in just 45 seconds(7k iterations). In the original Gaussian splatting papers, achieving a test PSNR of 18.892 and 23.506 at 7K resolution was considered good on train and truck, respectively. Our results are comparable to these benchmarks.

### Reproduction Instructions

Set up the [Tanks&Temple and DeepBlending Dataset](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) and Python environment as outlined previously, then execute the following:
```bash
   bash examples/train_truck_1k/eval_train_truck_1k.sh <path_to_save_experiment_results> <path_to_tandb_dataset>
   ```

(TODO: check these scripts have no side-effects)

## Experimental Setup for all experiments statistics above

- **Hardware**: 4x 40GB NVIDIA A100 GPUs
- **Interconnect**: Fully-connected Bidirectional 25GB/s NVLINK

---



# New features [Please check regularly!]

- We will release our optimized cuda kernels within gaussian splatting soon for further speed up. 
- We will support gsplat later as another choice of our cuda kernel backend. 

# Paper and Citation

Our system design, analysis of large-batch training dynamics, and insights from scaling up are all documented in the paper below: 

> [**On Scaling Up 3D Gaussian Splatting Training**](https://arxiv.org/abs/2406.18533)<br>
> [**Hexu Zhao¹**](https://tarzanzhao.github.io), [**Haoyang Weng¹\***](https://egalahad.github.io), [**Daohan Lu¹\***](https://daohanlu.github.io), [**Ang Li²**](https://www.angliphd.com), [**Jinyang Li¹**](https://www.news.cs.nyu.edu/~jinyang/), [**Aurojit Panda¹**](https://cs.nyu.edu/~apanda/), [**Saining Xie¹**](https://www.sainingxie.com)  (\* *co-second authors*)
> <br>¹New York University, ²Pacific Northwest National Laboratory <br>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@misc{zhao2024scaling3dgaussiansplatting,
      title={On Scaling Up 3D Gaussian Splatting Training}, 
      author={Hexu Zhao and Haoyang Weng and Daohan Lu and Ang Li and Jinyang Li and Aurojit Panda and Saining Xie},
      year={2024},
      eprint={2406.18533},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.18533}, 
}</code></pre>
  </div>
</section> 

# Code Specification
Please use "black" with default settings to format the code if you want to contribute.
## Setup
```shell
conda install black==24.4.2
```

# License

Distributed under the Apache License Version 2.0 License. See `LICENSE.txt` for more information.

# Reference

1. Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, July 2023. URL: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/.
