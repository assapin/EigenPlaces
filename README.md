
# EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition

This is the official pyTorch implementation of the ICCV 2023 paper "EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition".
The paper presents a new training method which aims at providing samples from multiple viewpoints to the model, to make it robust to camera viewpoint changes. It achieves SOTA on any dataset with large viewpoint shifts between query images and database.

For the paper we also released a codebase to reproduce results with all other baselines (NetVLAD, SFRS, Conv-AP, CosPlace, MixVPR) in order to have a standardized and fair evaluation framework at [https://github.com/gmberton/VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation)

[[ICCV 2023 Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Berton_EigenPlaces_Training_Viewpoint_Robust_Models_for_Visual_Place_Recognition_ICCV_2023_paper.html)] [[ArXiv](https://arxiv.org/abs/2308.10832)] [[BibTex](https://github.com/gmberton/EigenPlaces#cite)]

<p float="left">
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/assets/EigenPlaces/teaser.jpg" height="150" />
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/assets/EigenPlaces/eigen_map.jpg" height="150" /> 
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/assets/EigenPlaces/lateral_loss.png" height="150" />
  <img src="https://github.com/gmberton/gmberton.github.io/blob/main/assets/EigenPlaces/frontal_loss.png" height="150" />
</p>


## Train
Training is performed on the SF-XL dataset, which you can download from [here](https://github.com/gmberton/CosPlace). Make sure to download the training panoramas, which EigenPlaces takes as input and automatically crops with the required orientation.
After downloading the SF-XL dataset, simply run 

`$ python3 train.py --train_dataset_folder path/to/sf_xl/raw/train/panoramas --val_dataset_folder path/to/sf_xl/processed/val --test_dataset_folder path/to/sf_xl/processed/test `

the script automatically splits SF-XL in CosPlace Groups, and saves the resulting object in the folder `cache`.
By default training is performed with a ResNet-18 with descriptors dimensionality 512 and AMP, which uses less than 8GB of VRAM.

To change the backbone or the output descriptors dimensionality simply run something like this

`$ python3 train.py --backbone ResNet50 --fc_output_dim 128`

Run `$ python3 train.py -h` to have a look at all the hyperparameters that you can change. You will find all hyperparameters mentioned in the paper.

## Test
You can test a trained model as such

`$ python3 eval.py --backbone ResNet50 --fc_output_dim 128 --resume_model path/to/best_model.pth`

You can download plenty of trained models below.

## Trained Models

We have all our trained models on [PyTorch Hub](https://pytorch.org/docs/stable/hub.html), so that you can use them in any codebase without cloning this repository simply like this
```
import torch
model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048)
```

## running with dino

```
--device cuda \
--train_dataset_folder /Users/assaf/projects/fingerpic/data/small/train \
--test_dataset_folder /Users/assaf/projects/fingerpic/data/small/test \
--val_dataset_folder /Users/assaf/projects/fingerpic/data/small/val \
--backbone dinov2_vitb14 \
--image_size_dimension 224 \
--batch_size 48 \
--epochs_num 1 \ 
--iterations_per_epoch 300 

```


## Acknowledgements
Parts of this repo are inspired by the following repositories:
- [CosFace implementation in PyTorch](https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py)
- [CNN Image Retrieval in PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) (for the GeM layer)
- [Visual Geo-localization benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) (for the evaluation / test code)
- [CosPlace](https://github.com/gmberton/EigenPlaces)

## Cite
Here is the bibtex to cite our paper
```
@inproceedings{Berton_2023_EigenPlaces,
  title={EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition},
  author={Berton, Gabriele and Trivigno, Gabriele and Caputo, Barbara and Masone, Carlo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023},
  month={October},
  pages={11080-11090}
}
```
