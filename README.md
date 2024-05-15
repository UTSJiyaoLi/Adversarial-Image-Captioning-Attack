# AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization

## Overview
This is a Python implementation of ``AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization''. A GPU environment is required for running the code.

## Requirements:
The code is mainly written in Python 3.8.
> rouge-score==0.1.2 \\
> tensorflow==2.7.4 \\
> torch==2.0.1 \\

```
Command line:
$ pip install -r requirements.txt
```
To run the attack,
First, you need to download the model from the following links:

SAT: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

BLIP: https://github.com/salesforce/BLIP

Secondly, the datasets are available at:

COCO -- https://cocodataset.org/#home
Flickr8k -- https://www.kaggle.com/datasets/adityajn105/flickr8k
