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
import matplotlib


matplotlib.use('Agg')  # Or try other backends like 'TkAgg', 'QtAgg', etc.
# Create the parser
parser = argparse.ArgumentParser(description='The arguments')
parser.add_argument('--name', type=str, required=False, default='flicker8k', help='Dataset name')
parser.add_argument('--num_input', type=int, required=False, default=1000, help='number of input data.')
parser.add_argument('--pixels', type=int, required=False, default=500, help='Number of pixels to attack.')
parser.add_argument('--max_iter', type=int, required=False, default=5, help='Max iteration of DE algorithm.')
parser.add_argument('--pop_size', type=int, required=False, default= 20, help='Population size.')
parser.add_argument('--keywords', type=int, required=False, default= 50, help='Number of keywords')
parser.add_argument('--F', type=int, required=False, default=0.5, help='Mutation scale factor.')
parser.add_argument('--image_size', type=int, required=False, default=255, help='Input image size.')
parser.add_argument('--metric', type=str, required=False, default='bleu', help='metrics used for comparing attack performance.')
parser.add_argument('--separate', type=bool, required=False, default=False, help='Whether to attack keywords separately.')
parser.add_argument('--attention', type=bool, required=False, default=True, help='Whether use attention?')
parser.add_argument('--save_img', type=bool, required=False, default=False, help='Save the perturbed image?')
args = parser.parse_args()


def display(id, pixel, notes):
    # Read raw image
    with Image.open(image_path(id)) as image:
        image = image.resize((255,255))
        image_array = np.array(image)
        # Color to red
        for x, y in pixel:
            image_array[y, x, 0] = 255
        modified_image = Image.fromarray(image_array)
        modified_image.save(f"outputs/flicker8k_500p_1000_overall_255/{id}_{notes}.png")
    return 'Finished!'


def attack(num_input=args.num_input, 
           pixels=args.pixels,
           keywords=args.keywords,
           attention=args.attention,
           save_img=args.save_img,
           ):
    time_1 = time.time()
    # Get random sample.
    random.seed(42)
    ids = random.sample(range(8000), num_input)
    results = []
    
    attention_value = AttentionValue(num_input, keywords, 0.5, dir='attens_flicker8k_1000samples_255.json', percentage=True)
    index = attention_value.__get_item__(False, 'separate', pick_method='topk')
    index_allin = attention_value.__get_item__(False, 'all-in', pick_method='topk')
    print('Collected all attentions')

    for i in range(num_input):
        if i > int(num_input * 0.9):
            save_img = True
        # Get path
        folder_path = '/home/jiyli/Data/Image_Attack/data/train2014'
        files = os.listdir(folder_path)
        files = sorted(files)
        image_path = folder_path + '/' + str(files[ids[i]])
        
        # Read image and process
        img = imread(image_path)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = imresize(img, (255, 255))
        img = img.transpose(2, 0, 1)
        img = img / 255.
        img = torch.FloatTensor(img).to(device)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([normalize])
        image = transform(img)  # (3, image_size, image_size)

        # BLEU score before attack
        words, alphas, seq = sat(image)
        tgt = get_coco_dataset_for_sat(ids[i])[1]
        bleu_original = bleu4(tgt, words)

        # Different attack setting 
        _, best_att_sep, best_pixels_att_sep, caption_att_sep = attack_sat(ids[i], pixels, index[i], 50, condition='separate', save_img=save_img,dataset='flicker8k')
        _, best_noatt, best_pixels_noatt, caption_noatt = attack_sat(ids[i], pixels, index[i], 50, condition='no_att', save_img=save_img,dataset='flicker8k')
        _, best_att_allin, best_pixels_att_allin, caption_att_alllin = attack_sat(ids[i], pixels, index_allin[i], 50, condition='all-in', save_img=save_img, dataset='flicker8k')

        # Collecting results
        results.append([ids[i], (words, bleu_original), (caption_att_sep, best_att_sep), 
        (caption_noatt, best_noatt), (caption_att_alllin, best_att_allin)])
        
        # Saving images
        if save_img:
            print(f'Start saving image {i}')
            display(ids[i], index_allin[i], notes='allin')
            display(ids[i], index[i], notes='separate')
            display(ids[i], best_pixels_att_sep, notes='att_sep')
            display(ids[i], best_pixels_noatt, notes='noatt')
            display(ids[i], best_pixels_att_allin, notes='att_allin')

    # Calculate average score of each method.
    best_allin, bleurouge_diff_allin, embeddings_score_diff_allin, bleu1_diff_allin, bleu2_diff_allin, rouge_diff_1_allin, rouge_diff_2_allin, f_original_allin, bleu1_orig_allin, bleu2_orig_allin, rouge1_orig_allin, rouge2_orig_allin = 0,0,0,0,0,0,0,0,0,0,0,0
    best_sep, bleurouge_diff_sep, embeddings_score_diff_sep, bleu1_diff_sep, bleu2_diff_sep, rouge_diff_1_sep, rouge_diff_2_sep, f_original_sep, bleu1_orig_sep, bleu2_orig_sep, rouge1_orig_sep, rouge2_orig_sep = 0,0,0,0,0,0,0,0,0,0,0,0
    best_noatt_, bleurouge_diff_noatt, embeddings_score_diff_noatt, bleu1_diff_noatt, bleu2_diff_noatt, rouge_diff_1_noatt, rouge_diff_2_noatt, f_original_noatt, bleu1_orig_noatt, bleu2_orig_noatt, rouge1_orig_noatt, rouge2_orig_noatt = 0,0,0,0,0,0,0,0,0,0,0,0
    allin, sep, noatt = [], [], []
    dev_allin, dev_sep, dev_noatt = [], [], []

    # Standard Deviation
    for j in results:
        dev_allin.append(j[4][1][0])
        dev_sep.append(j[2][1][0])
        dev_noatt.append(j[3][1][0])
    std_dev_allin = std_dev(dev_allin)
    std_dev_sep = std_dev(dev_sep)
    std_dev_noatt = std_dev(dev_noatt)
    
    for i in results:
        allin.append(i[4][1])
        sep.append(i[2][1])
        noatt.append(i[3][1])
    for i in allin:
        best_allin += i[0]
        bleurouge_diff_allin += i[1]
        embeddings_score_diff_allin += i[2]
        bleu1_diff_allin += i[3]
        bleu2_diff_allin += i[4]
        rouge_diff_1_allin += i[5]
        rouge_diff_2_allin += i[6]
        f_original_allin += i[7]
        bleu1_orig_allin += i[8]
        bleu2_orig_allin += i[9] 
        rouge1_orig_allin += i[10] 
        rouge2_orig_allin += i[11]
    for i in sep:
        best_sep += i[0]
        bleurouge_diff_sep += i[1]
        embeddings_score_diff_sep += i[2]
        bleu1_diff_sep += i[3]
        bleu2_diff_sep += i[4]
        rouge_diff_1_sep += i[5]
        rouge_diff_2_sep += i[6]
        f_original_sep += i[7]
        bleu1_orig_sep += i[8]
        bleu2_orig_sep += i[9] 
        rouge1_orig_sep += i[10] 
        rouge2_orig_sep += i[11]
    for i in noatt:
        best_noatt_ += i[0]
        bleurouge_diff_noatt += i[1]
        embeddings_score_diff_noatt += i[2]
        bleu1_diff_noatt += i[3]
        bleu2_diff_noatt += i[4]
        rouge_diff_1_noatt += i[5]
        rouge_diff_2_noatt += i[6]
        f_original_noatt += i[7]
        bleu1_orig_noatt += i[8]
        bleu2_orig_noatt += i[9] 
        rouge1_orig_noatt += i[10] 
        rouge2_orig_noatt += i[11]

    overall_results = {}

    overall_results['allin'] = (round(best_allin / num_input, 4),
                                round(bleurouge_diff_allin / num_input, 4), round(embeddings_score_diff_allin / num_input,4),
                                round(bleu1_diff_allin / num_input, 4), round(bleu2_diff_allin / num_input, 4), 
                                round(rouge_diff_1_allin / num_input , 4), round(rouge_diff_2_allin / num_input, 4), 
                                round(f_original_allin / num_input, 4), round(std_dev_allin, 4),
                                round(bleu1_orig_allin / num_input, 4), round(bleu2_orig_allin / num_input, 4),
                                round(rouge1_orig_allin / num_input, 4), round(rouge2_orig_allin / num_input, 4))
    overall_results['sep'] = (round(best_sep / num_input, 4), 
                                round(bleurouge_diff_sep / num_input, 4), round(embeddings_score_diff_sep / num_input, 4),
                                round(bleu1_diff_sep / num_input, 4), round(bleu2_diff_sep / num_input, 4), 
                                round(rouge_diff_1_sep / num_input , 4), round(rouge_diff_2_sep / num_input, 4), 
                                round(f_original_sep / num_input, 4), round(std_dev_sep, 4),
                                round(bleu1_orig_sep / num_input, 4), round(bleu2_orig_sep / num_input, 4),
                                round(rouge1_orig_sep / num_input, 4), round(rouge2_orig_sep / num_input, 4))
    overall_results['noatt'] = (round(best_noatt_ / num_input, 4),
                                round(bleurouge_diff_noatt / num_input, 4), round(embeddings_score_diff_noatt / num_input, 4),
                                round(bleu1_diff_noatt / num_input, 4), round(bleu2_diff_noatt / num_input, 4), 
                                round(rouge_diff_1_noatt / num_input , 4), round(rouge_diff_2_noatt / num_input, 4), 
                                round(f_original_noatt / num_input, 4), round(std_dev_noatt, 4),
                                round(bleu1_orig_noatt / num_input, 4), round(bleu2_orig_noatt / num_input, 4),
                                round(rouge1_orig_noatt / num_input, 4), round(rouge2_orig_noatt / num_input, 4))

    # Dumping
    with open("outputs/jsons/flicker8k_500p_1000_overall_255.json", "wb") as file:
        pickle.dump((overall_results, results), file)  

    # Calculate time
    time_2 = time.time()
    time_cus = convert_to_preferred_format(time_2 - time_1)
    print('time: ', time_cus)

    return 'finish dumping!'

# nohup python -u main.py>outputs/logger_Att_1000p_100samples.txt 2>&1 &

if __name__ == '__main__':
    attack()