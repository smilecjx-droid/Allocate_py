# ALLOCATE ： **A**daptive **L**earning of optimaL-transp**O**rt-based **C**ell **A**lignmen**T** and **E**volution

![Fig1](https://github.com/user-attachments/assets/2bbf3a6a-45c9-4682-9d8f-402f2f4e8b44)


ALLOCATE advances optimal transport by explicitly modeling key characteristics of temporally evolving cellular systems, improving the corresponding assignment probabilities. It also provides interpretability for the potential of cellular growth and death, enabling a quantitative description of developmental and disease dynamics.

The main inputs are:

- molecular profiles of cells or spots at two consecutive time points;
- optional spatial coordinates for spatially resolved data;
- user-defined parameters controlling the transport optimization.

The main outputs are:

- an optimal transport matrix between source and target cells;
- an adaptively inferred source marginal;
- optional diagnostic information and evaluation metrics.


## Installation

This repository currently provides research code rather than a fully packaged Python library. The code can be used by cloning the repository and installing the required dependencies.

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ALLOCATE.git
cd ALLOCATE
```

We recommend creating a clean conda environment:

```bash
conda create -n allocate python=3.10
conda activate allocate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

We will soon make ALLOCATE available on PyPi. In the mean time, you can download the repository and call the functions directly.

## Contact
If you encounter any problem running the software, please contact Jiaxin Chen at cjx_bio@sjtu.edu.cn

## Tutorials
For a step-by-step tutorial, please refer to the [vignette](vignette/) directory in the repository, which provides usage examples.

## Reference

