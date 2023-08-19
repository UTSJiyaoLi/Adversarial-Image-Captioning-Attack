import numpy as np
import random
import pickle
import time
import argparse
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import *
from metrics import *
if sys.version_info.minor == 8:
    from blip_colab import blip
    from blip_model import *
    from perturb import differential_evolution
    from perturb_att import differential_evolution_att
from coco import get_coco_dataset
from dumping import *
from dataloader import dataloader
from pick_atten import *
from attack_sat import *

# Create the parser
parser = argparse.ArgumentParser(description='The arguments')
parser.add_argument('--name', type=str, required=False, default='coco', help='Dataset name')
parser.add_argument('--num_input', type=int, required=False, default=100, help='number of input data.')
parser.add_argument('--pixels', type=int, required=False, default=300, help='Number of pixels to attack.')
parser.add_argument('--max_iter', type=int, required=False, default=5, help='Max iteration of DE algorithm.')
parser.add_argument('--pop_size', type=int, required=False, default= 20, help='Population size.')
parser.add_argument('--keywords', type=int, required=False, default= 50, help='Number of keywords')
parser.add_argument('--F', type=int, required=False, default=0.5, help='Mutation scale factor.')
parser.add_argument('--image_size', type=int, required=False, default=384, help='Input image size.')
parser.add_argument('--metric', type=str, required=False, default='bleu', help='metrics used for comparing attack performance.')
parser.add_argument('--separate', type=bool, required=False, default=False, help='Whether to attack keywords separately.')
parser.add_argument('--attention', type=bool, required=False, default=True, help='Whether use attention?')
parser.add_argument('--save_img', type=bool, required=False, default=False, help='Save the perturbed image?')
args = parser.parse_args()

def attack(num_input=args.num_input, 
           pixels=args.pixels,
           keywords=args.keywords,
           attention=args.attention,
           save_img=args.save_img,
           ):
    time_1 = time.time()
    # Get random sample.
    random.seed(42)
    ids = random.sample(range(80000), num_input)
    if not attention:
        results, diffs = {}, {}
        diff = []
        attention_value = AttentionValue(num_input, keywords, pixels, dir='attens_100samples.json', percentage=False)
        index = attention_value.__get_item__(False, 'separate', pick_method='topk')
        for i in range(num_input):
            if i > int(0.9 * num_input):
                save_img = True
            _, best, _, _ = attack_sat(ids[i], pixels, index=indexes, condition='no_att', keywords=50, save_img=save_img)
            diff.append(best)
            results[('id: ', ids[i])] = best
        diffs = ('max: ', max(diff), 'average: ', sum(diff)/len(diff))

        file = open("outputs/noatt_500p_100samples_sat.json", "wb")
        pickle.dump((results, diffs), file)
        file.close()
    
    if attention:
        results = {}
        diff = []
        diffs = {}
        attention_value = AttentionValue(num_input, keywords, pixels, dir='attens_100samples.json', percentage=False)
        index = attention_value.__get_item__(False, 'all-in', pick_method='topk')
        for i in range(num_input):
            if i > int(0.9 * num_input):
                save_img = True
            _, best, _, _ = attack_sat(ids[i], pixels, index[i], 50, condition='all-in', save_img=save_img)
            results[('id: ', ids[i])] = best
            diff.append(best)
        diffs = ('max: ', max(diff), 'average: ', sum(diff)/len(diff))

        # Dumping
        file = open("outputs/att_100samples_sat_top300_allin.json", "wb")
        pickle.dump((results, diffs), file)  # length = k * len(ids)
        file.close()

    # Calculate time
    time_2 = time.time()
    time_cus = convert_to_preferred_format(time_2 - time_1)
    print('time: ', time_cus)

    return 'finish dumping!'

# nohup python -u main.py>outputs/logger_Att_1000p_100samples.txt 2>&1 &

if __name__ == '__main__':
    # Check 3 things before run the code:
    # first logger directory and name.
    # Second, json file name and directory
    # Third, save_image folder in <utils.py>, and the 'is_attention' label.
    attack()
