"""
Hints:

Cannot Use the lavis toolkits. It requires transformers
salesforce-lavis 1.0.2 requires transformers<4.27,>=4.25.0, but you have transformers 4.15.0 which is incompatible.

For more information on: https://github.com/salesforce/LAVIS

To apply Lavis uncomment 'import lavis.models' and line 16-17 after install proper version of transformers.

from lavis.models import load_model_and_preprocess

"""

import torch
import sys
sys.path.insert(1, 'BLIP')
from models.blip import blip_decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BLIP pretrained model
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
model = blip_decoder(pretrained=model_url, image_size=384, vit='base')

# Show attend and tell pre-trained model om COCO 2014 ddataset
# For more information go: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/tree/master

# checkpoint_path = 'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
# checkpoint = torch.load(checkpoint_path, map_location=str(device))
# # print(f'checkpoint: {checkpoint}')
# decoder = checkpoint['decoder']
# # attention = checkpoint['attention']
# decoder = decoder.to(device)
# decoder.eval()
# encoder = checkpoint['encoder']
# encoder = encoder.to(device)
# encoder.eval()


if __name__ == '__main__':
    print('model: ', model)