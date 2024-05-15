# AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization

## Overview
This is a Python implementation of ``AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization''. A GPU environment is required for running the code.

<img width="679" alt="github_exmaple" src="https://github.com/UTSJiyaoLi/Adversarial-Image-Captioning-Attack/assets/49722565/df6d5611-541c-4af9-9e40-74e1069303dc">

## Requirements:
The code is mainly written in Python 3.8.
> rouge-score==0.1.2
> 
> tensorflow==2.7.4
> 
> torch==2.0.1
>

Other Python packages can be installed by running the following command from the command line.

```
Command line:
$ pip install -r requirements.txt
```

## Set up
First, you need to download victim models from the following links:

SAT: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

BLIP: https://github.com/salesforce/BLIP

Secondly, the datasets are available at:

COCO -- https://cocodataset.org/#home
Flickr8k -- https://www.kaggle.com/datasets/adityajn105/flickr8k

## Evaluation
To attack an image, simply run the `` attack_sat.py", or ``attack_blip_test.py" according to the victim model.
For example, to attack SAT:

```
$ python3 attack_sat.py
```

