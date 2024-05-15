import numpy as np
import random
import argparse
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from PIL import Image
from utils import *
from metrics import *
from sat import *
from coco import get_coco_dataset
from dumping import *
from dataloader import dataloader
from pick_atten import *
from sat import *
import warnings

warnings.filterwarnings('ignore')
# Hints:
# Between flicker8k and coco
# we need to mod few things:
#     1. dataset name 
#     2. random ids number and id in per run 
#     3. attention file name 
#     4.
def attack_sat(id, pixels, index, keywords, pick_method='topk', metric='bleu', condition='all-in', image_size=255, max_iter=5, pop_size=20, F=0.5, save_img=False, dataset='coco'):
    """
    id: image id
    pixels: actual number of pixels you want to attack.
    keywords: number of keywords you want to attack.
    pick_method: 'topk' or 'withweight'
    metric: the metric used to evaulate. 'bleu' or 'rouge'
    condition: the way to use attention, 'separate' or 'all-in' see file 'pick_atten.py'.
    image_size=384, 
    max_iter=5, 
    pop_size=20, 
    F=0.5, 
    save_img: save the perturbed image?
    """

    # Initialize population
    # Initialize pixel position according to attention
    if condition == 'all-in':
        # In the all-in mode, we only randomly choose pixels from index that attention give to us.
        if pick_method == 'topk':
            indexes = random.sample(index, pixels)
            remaining_index = [num for num in indexes if num not in index]

        if pick_method == 'withweight':
            indexes = index

        bounds = np.array([(0, 254), (0, 254), (0, 255), (0, 255), (0, 255)] * pixels)    # (x, y, R, G, B) * pixels attacked
        pop = np.zeros((pop_size, len(bounds)))
        for i in range(pop_size):
            for j in range(pixels):
                pop[i][5 * j] = indexes[j][0]
                pop[i][5 * j + 1] = indexes[j][1]
                pop[i][5 * j + 2] = random.randint(-40, 40)
                pop[i][5 * j + 3] = random.randint(-40, 40)
                pop[i][5 * j + 4] = random.randint(-40, 40)
    
    if condition == 'separate':
        # In the separate mode, we pick top-k for each word and pick from whole set.
        indexes = random.sample(index, pixels)
        bounds = np.array([(0, 254), (0, 254), (0, 255), (0, 255), (0, 255)] * pixels)    # (x, y, R, G, B) * pixels attacked
        pop = np.zeros((pop_size, len(bounds)))
        for i in range(pop_size):
            for j in range(len(indexes)):
                pop[i][5 * j] = indexes[j][0]
                pop[i][5 * j + 1] = indexes[j][1]
                pop[i][5 * j + 2] = random.randint(-40, 40)
                pop[i][5 * j + 3] = random.randint(-40, 40)
                pop[i][5 * j + 4] = random.randint(-40, 40)

    if condition == 'no_att':
        bounds = np.array([(0, 254), (0, 254), (0, 255), (0, 255), (0, 255)] * pixels)    # (x, y, R, G, B) * pixels attacked
        pop = np.zeros((pop_size, len(bounds)))
        for i in range(pop_size):
            for j in range(len(bounds)):
                if 1 <= (j+1) % 5 <= 2:
                    pop[i][j] = random.uniform(bounds[j][0], bounds[j][1])
                else:
                    pop[i][j] = random.randint(-40, 40)

    # Form a x_0 from initialized poplation.     
    x_0 = []  # [image, pixel]
    for pops in pop:
        pops = [int(i) for i in pops]
        pix = divide_list(pops, pixels)
        _, ori_img = dataloader.get_item_sat(dataset, id, image_size)
        for num_pixels in range(pixels):
            for m in range(3):
                ori_img[m][pix[num_pixels][0]][pix[num_pixels][1]] = pix[num_pixels][2 + m] + ori_img[m][pix[num_pixels][0]][pix[num_pixels][1]]
        x_0.append([ori_img, pix])

    # Functions that used to prepare image.
    
    # Set a transform  <image to [-1, 1]>
    transform_tensor = transforms.Compose([
                    transforms.Normalize
                    ((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225))
                    ])

    # Get target image (for getting reference captions)
    if dataset =='coco':
        ref_caption = get_coco_dataset_for_sat(id)[1]
    if dataset == 'flicker8k':
        ref_caption = get_flickr8k_dataset_for_sat(id)[1]

    # Iterate until maximum number of iterations reached
    x_0_score = []
    count = []
    for i in range(max_iter):
        count_1 = []
        for j in range(pop_size):
            img, new_img = dataloader.get_item_sat(dataset, id, image_size)
            print(f'\n id {id}')
            # print('f_pre: ', bleu4(ref_caption, sat(image = transform_tensor(torch.FloatTensor(x_0[j][0]/384.)).to(device))[0]))
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
                    new_img[m][cand[num_pixels][0]][cand[num_pixels][1]] = cand[num_pixels][2 + m] + new_img[m][cand[num_pixels][0]][cand[num_pixels][1]]
            # print("New image pixel: ", new_img[0][cand[1][0]][cand[1][1]])
            
            # Transform the PIL image to tensor [1, 3, W, H] within [-1, 1]
            new_img_ = transform_tensor(torch.FloatTensor(new_img/255.)).to(device) # Get perturbed image for gen i pop j
            x_0_img = transform_tensor(torch.FloatTensor(x_0[j][0]/255.)).to(device)
            f_original = bleu4(ref_caption, sat(image = img)[0])       # Original
            f_pre = bleu4(ref_caption, sat(image = x_0_img)[0])       # Previous generation 
            f_mutant = bleu4(ref_caption, sat(image = new_img_)[0])   # This generation
            print('tgt: ', ref_caption)
            print('new: ', sat(image = new_img_)[0])
            print('old: ', sat(image = img)[0])
            print(f_mutant, f_original)

            # Replace current population member with mutant vector if mutant vector is better
            if f_mutant < f_original: # Success attack
                if f_mutant < f_pre: # Outperform than previous generation
                    print(f_mutant, f_pre)
                    x_0[j] = [new_img, cand]
                count_1.append(f_original - f_mutant)
            count_1.append([])
        count.append(count_1)

    # Get the best x_0 from x_0 list
    for n in range(len(x_0)):
        item = transform_tensor(torch.FloatTensor(x_0[n][0]/255.)).to(device)
        x_0_score.append(bleu4(ref_caption, sat(image=item)[0]))
    
    min_index, min_value = min(enumerate(x_0_score), key=lambda x: x[1])

    print('f_original, min_value', f_original, min_value)

    # Calculate the largest (best) bleu score and its corresponding outputs.
    best = f_original - min_value
    best_img = x_0[min_index][0]
    pixels_and_values = x_0[min_index][1]
    best_pixels = []
    for i in pixels_and_values:
        best_pixels.append((i[0], i[1]))
    img_after_att = transform_tensor(torch.FloatTensor(best_img/255.)).to(device)
    cap_after_attack = sat(image=img_after_att)[0]

    # Gathering metrics
    # rouge 1, 2
    rouge_orig = rouge(ref_caption, sat(image = img)[0])
    rouge_score_aft = rouge(ref_caption, cap_after_attack)
    rouge1_diff = rouge_orig[0] - rouge_score_aft[0]
    rouge2_diff = rouge_orig[1] - rouge_score_aft[1]
    # bleu + rouge
    bleurouge_orig = bleurouge(f_original, rouge_orig[0])
    bleurouge_aft = bleurouge(min_value, rouge_score_aft[0])
    bleurouge_diff = bleurouge_orig - bleurouge_aft
    # bleu 1
    bleu1_orig = bleu1(ref_caption, sat(image = img)[0])
    bleu1_aft = bleu1(ref_caption, cap_after_attack)
    bleu1_diff = bleu1_orig - bleu1_aft
    # bleu 2
    bleu2_orig = bleu2(ref_caption, sat(image = img)[0])
    bleu2_aft = bleu2(ref_caption, cap_after_attack)
    bleu2_diff = bleu2_orig - bleu2_aft
    # word embeddings cosine
    embeddings_score_orig, embeddings_score_aft = 0, 0
    for sent in ref_caption:
        embeddings_score_orig += word_embeddings(sent, sat(image = img)[0])
    embeddings_score_orig = embeddings_score_orig / len(ref_caption)
    for sent in ref_caption:
        embeddings_score_aft += word_embeddings(sent, cap_after_attack)
    embeddings_score_aft = embeddings_score_aft / len(ref_caption)
    embeddings_score_diff = embeddings_score_orig - embeddings_score_aft

    # Saving perturbed image.
    if save_img:
        save_image(id, pixels, best_img, is_attention=condition)

    return count, (best, bleurouge_diff, embeddings_score_diff, bleu1_diff, bleu2_diff, rouge1_diff, rouge2_diff, f_original, bleu1_orig, bleu2_orig, rouge_orig[0], rouge_orig[1]), best_pixels, cap_after_attack


if __name__ == '__main__':

    keywords, num_input, pixels = 50, 1000, 500
    id = 48

    # Get position of image in attention files
    random.seed(42)
    ids = random.sample(range(8000), num_input)
    pos = ids.index(id)

    # Get attention
    attention_value = AttentionValue(num_input, keywords, 0.5, dir='attens_flicker8k_1000samples_255.json', percentage=True)
    index = attention_value.__get_item__(False, 'all-in', pick_method='topk')[pos]
    index_sep = attention_value.__get_item__(False, 'separate', pick_method='topk')[pos]
    print('Got attentions!')
    # Run attack
    # _, best_att, best_pixels_att, caption_att = attack_sat(id, pixels, index, 50, condition='separate')
    # _, best_noatt, best_pixels_noatt, caption_noatt = attack_sat(id, pixels, index, 50, condition='no_att')
    _, best_allin, best_pixels_att_allin, caption_att_allin = attack_sat(id, pixels, index, 50, condition='all-in')
    # _, best_separate, best_pixels_att_sep, caption_att_sep = attack_sat(id, pixels, index_sep, 50, condition='separate')
    # _, best_noatt, best_pixels_att_noatt, caption_att_noatt = attack_sat(id, pixels, index, 50, condition='no_att')

    print(caption_att_allin, caption_att_sep, caption_att_noatt)
