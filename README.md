# Machine Learning Models

This folder contains code for training the machine learning models AerialFormer, PyramidMamba, SFA-Net and UNetFormer on the OpenEarthMap semantic segmentation training set, and for inferring PyramidMamba and calculating the imperviousness near trees from semantic masks.

## Inference with PyramidMamba

The following section gives an overview on how to run inference with the PyramidMamba model.

*Notice: The model weights cannot be provided as they are too large for Git.*

*Notice: All commands are run from the root directory `master-ml-models`.*

### Installation

Open the folder **master-ml-models** using a terminal and create a python environment:

```
conda create -n mmm python=3.8
conda activate mmm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r GeoSeg/requirements.txt
```

Install Mamba
```
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
```

### Inference

Make sure you create folders for your aerial imagery, your masks and the output of the imperviousness calculation.

Add the desired aerial imagery to the respective folder. Then execute the following command:

```
python3 GeoSeg/inference_huge_image_alt.py -i my_images/ -o my_masks/ -c GeoSeg/config/oem/pyramidmamba-inf.py
```

Afterwards, run the following command to calculate the imperviousness near trees:

```
python3 imperviousness_calculation.py -m my_masks/ -o imp/
```

## Acknowledgment

This work is based on the GeoSeg project. Please see `GeoSeg/` for more details. Thank you Dr. Libo Wang!

The machine learning models referenced are based on these publications:

- [AerialFormer](https://arxiv.org/abs/2306.06842)
- [PyramidMamba](https://arxiv.org/abs/2406.10828)
- [SFA-Net](https://doi.org/10.3390/rs16173278)
- [UNetFormer](https://arxiv.org/abs/2109.08937)

Many thanks to the following projects for their contribution to machine learning:

- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

