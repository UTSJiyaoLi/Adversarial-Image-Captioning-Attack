# Official processing for CoCo dataset.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision.datasets as dset
import torchvision.transforms as transforms
import sys,os
if sys.version_info.minor == 6:
    from scipy.misc import imread, imresize
import json
from PIL import Image
# from torchvision.transforms.functional import InterpolationMode
# from torch.nn.functional import interpolate # torch 2.0.1

# cap_ = dset.CocoCaptions(root = '/home/jiyli/Data/Image_Attack/data/test2014',
#                         annFile = '/home/jiyli/Data/Image_Attack/data/annotations/image_info_test2014.json',
#                         transform=transforms.Compose([
#                         transforms.Resize(size=(255, 255))]))
cap = dset.CocoCaptions(root = '/home/jiyli/Data/Image_Attack/data/train2014',
                        annFile = '/home/jiyli/Data/Image_Attack/data/annotations/captions_train2014.json',
                        transform=transforms.Compose([
                        transforms.Resize(size=(255, 255))]))
# cap_flicker8k = dset.Flickr8k(root = '/home/jiyli/Data/Image_Attack/captionattack/Flicker8k_Dataset',
#                         ann_file = '/home/jiyli/Data/Image_Attack/captionattack/Flickr8k_text/Flickr8k.token.txt',
#                         transform=transforms.Compose([
#                         transforms.Resize(size=(255, 255))]))


def get_coco_dataset(id=3, for_attention=False):
    # Open the dataset, output image and captions.
    transform_totensor = transforms.ToTensor()
    img, target = cap[id]  # load n-th sample
    if for_attention:
        img = transform_totensor(img)
        return img
    else:
        return cap, target, img


def get_coco_dataset_for_sat(id, for_attention=False):
    pil_image, target = cap[id][0], cap[id][1]
    return cap, target, pil_image


def get_flickr8k_dataset_for_sat(id, for_attention=False):
    pil_image, target = cap_flicker8k[id][0], cap_flicker8k[id][1]
    return cap_flicker8k, target, pil_image
    return cap, target, pil_image


def get_flickr8k_dataset_for_sat(id):
    # Read captions file
    captions_file='/home/jiyli/Data/Image_Attack/captionattack/Flickr8k.token.txt'
    image_dir='/home/jiyli/Data/Image_Attack/captionattack/Flicker8k_Dataset'
    with open(captions_file, 'r') as f:
        captions_data = f.readlines()

    # Create a dictionary to store captions
    captions_dict = {}
    for line in captions_data:
        parts = line.strip().split('\t')
        image_name = parts[0].split('#')[0]  # Extract image name without #x
        caption = parts[1]
        if image_name not in captions_dict:
            captions_dict[image_name] = []
        captions_dict[image_name].append(caption)

    # Get the list of image IDs (filenames)
    image_ids = list(captions_dict.keys())

    # Check if the specified image number is valid
    if 0 <= id < len(image_ids):
        image_id = image_ids[id]
        captions = captions_dict[image_id]
        
        # Load and resize image
        image_file = os.path.join(image_dir, image_id)
        image = Image.open(image_file)
        image_resized = image.resize((255, 255), Image.ANTIALIAS)  # Resize to 255x255
        
        transform = transforms.Compose([transforms.ToTensor()])
        image_data = transform(image_resized)
        return captions, captions, image_data
    else:
        print("Invalid image number.")
        return [], None

def get_flickr8k_dataset(id):
    # Read captions file
    captions_file='/home/jiyli/Data/Image_Attack/captionattack/Flickr8k.token.txt'
    image_dir='/home/jiyli/Data/Image_Attack/captionattack/Flicker8k_Dataset'
    with open(captions_file, 'r') as f:
        captions_data = f.readlines()

    # Create a dictionary to store captions
    captions_dict = {}
    for line in captions_data:
        parts = line.strip().split('\t')
        image_name = parts[0].split('#')[0]  # Extract image name without #x
        caption = parts[1]
        if image_name not in captions_dict:
            captions_dict[image_name] = []
        captions_dict[image_name].append(caption)

    # Get the list of image IDs (filenames)
    image_ids = list(captions_dict.keys())

    # Check if the specified image number is valid
    if 0 <= id < len(image_ids):
        image_id = image_ids[id]
        captions = captions_dict[image_id]
        
        # Load and resize image
        image_file = os.path.join(image_dir, image_id)
        image = Image.open(image_file)
        image_resized = image.resize((384, 384), Image.ANTIALIAS)  # Resize to 255x255
        
        transform = transforms.Compose([transforms.ToTensor()])
        image_data = transform(image_resized)
        return captions, captions, image_data
    else:
        print("Invalid image number.")
        return [], None

if __name__ == '__main__':
    print(get_coco_dataset_for_sat(0)[2])
    print(flickr8k_dataset(5))
