import numpy as np
import random
import argparse
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import *
from blip_colab import blip
from metrics import *
from blip_model import *
from coco import get_coco_dataset
from dumping import *
from dataloader import *

# author = jiyli @ UTS

def differential_evolution(id, pixels, metric='bleu', image_size=384, max_iter=5, pop_size=20, F=0.5, save_img=False):
    
    # Initialize population
    bounds = np.array([(0, 383), (0, 383), (0, 255), (0, 255), (0, 255)] * pixels)    # (x, y, R, G, B) * pixels attacked
    pop = np.zeros((pop_size, len(bounds)))
    for i in range(pop_size):
      for j in range(len(bounds)):
        if 1 <= (j+1) % 5 <= 2:
            # print(pop[i][j])
            pop[i][j] = random.uniform(bounds[j][0], bounds[j][1])
        else:
            pop[i][j] = np.random.normal(0, 50)

    # Form an x_0 from initialized poplation.     
    x_0 = []  # [image, pixel]
    for pops in pop:
        pops = [int(i) for i in pops]
        pix = divide_list(pops, pixels)
        _, ori_img = dataloader.get_item('coco', id, image_size)
        for num_pixels in range(pixels):
            for m in range(3):
                # print(ori_img[m][pix[num_pixels][0]][pix[num_pixels][1]])
                ori_img[m][pix[num_pixels][0]][pix[num_pixels][1]] = pix[num_pixels][2 + m] + ori_img[m][pix[num_pixels][0]][pix[num_pixels][1]]
        x_0.append([ori_img, pix])
    # Functions that used to prepare image.
    # Set a transform  <image to [-1, 1]>
    transform_tensor = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize
            ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
   
    # Transform tensor [0, 255] to PIL image
    transform_toimage = transforms.ToPILImage()

    # Get target image (for getting reference captions)
    if metric == 'rouge':
        tgt = get_coco_dataset(id)[1][0]
    if metric == 'bleu':
        tgt = get_coco_dataset(id)[1]

    # Iterate until maximum number of iterations reached
    x_0_score, count = [], []
    for i in range(max_iter):
        count_1 = []
        for j in range(pop_size):
            img, new_img = dataloader.get_item('coco', id, image_size)
            print(f'\n id {id}')
            print('f_pre: ', bleu4(tgt, blip(image = transform_tensor(transform_toimage(x_0[j][0])).unsqueeze(0).to(device), model=model, image_size=image_size)))
            print(f'gen {i + 1} ---> pop: {j + 1}')
            # Choose three random population members
            candidates = random.sample(range(pop_size), 3)
            # Generate a mutant vector
            mutant = np.clip(pop[candidates[0]] + F * (pop[candidates[1]] - pop[candidates[2]]), bounds[:, 0], bounds[:, 1])

            # Modify the image
            mutant = [int(i) for i in mutant]
            if pixels < 2:
                mutant = [mutant]
            cand = divide_list(mutant, pixels)

            # print("Old image pixel: ", new_img[0][cand[1][0]][cand[1][1]])
            for num_pixels in range(pixels):
                for m in range(3):
                    new_img[m][cand[num_pixels][0]][cand[num_pixels][1]] = cand[num_pixels][2 + m]+ new_img[m][cand[num_pixels][0]][cand[num_pixels][1]]
            # print("New image pixel: ", new_img[0][cand[1][0]][cand[1][1]])
            
            # Transform the PIL image to tensor [1, 3, W, H] within [-1, 1]
            new_img_ = transform_tensor(transform_toimage(new_img)).unsqueeze(0).to(device) # Get perturbed image for gen i pop j
            x_0_img = transform_tensor(transform_toimage(x_0[j][0])).unsqueeze(0).to(device)
            f_original = bleu4(tgt, blip(image = img, model=model, image_size=image_size))       # Original
            f_pre = bleu4(tgt, blip(image = x_0_img, model=model, image_size=image_size))       # Previous generation 
            f_mutant = bleu4(tgt, blip(image = new_img_, model=model, image_size=image_size))   # This generation
            
            print('tgt: ', tgt)
            print('new: ', blip(image = new_img_, model=model, image_size=image_size))
            print('old: ', blip(image = img, model=model, image_size=image_size))
            print(f_mutant, f_original)

            # Replace current population member with mutant vector if mutant vector is better
            if f_mutant < f_original: # Success attack
                if f_mutant < f_pre: # Outperform than previous generation
                    # print(f_mutant, f_pre)
                    x_0[j] = [new_img, cand]
                count_1.append(f_original - f_mutant)
            count_1.append([])
        count.append(count_1)

    # Get the best x_0 from x_0 list
    for n in range(len(x_0)):
        item = transform_tensor(transform_toimage(x_0[n][0])).unsqueeze(0).to(device)
        x_0_score.append(bleu4(tgt, blip(image=item, model=model, image_size=image_size)))
    min_index, min_value = min(enumerate(x_0_score), key=lambda x: x[1])
    print(f_original, min_value)
    best = f_original - min_value
    best_img = x_0[min_index][0]

    # Saving perturbed image.
    if save_img:
        save_image(id, pixels, best_img)

    return count, best


if __name__ == '__main__':

    count, best = differential_evolution(29256, 700)
    print(best)