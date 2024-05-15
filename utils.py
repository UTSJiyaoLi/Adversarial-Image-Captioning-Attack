import torch
import os
import pickle
from PIL import Image, ImageDraw
import torchvision.datasets as dset
import torchvision.transforms as transforms
import sys
if sys.version_info.minor != 8:
    from scipy.misc import imread, imresize
if sys.version_info.minor == 8:
    from torchvision.transforms.functional import InterpolationMode  # for python 3.8, pytorch 2.0.1
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import requests
from coco import *
# from dataloader import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(id, image_size, device, before=True, dataset='coco'):

    '''
    input an pil.image the function will convert it into 
    image tensor range [0, 255] or [-1. 1]
    '''
    if dataset =='coco':
        _, _, raw_image = get_coco_dataset(id)
    if dataset =='flicker8k':
        _,_, raw_image = flickr8k_dataset(id)
    # Transform image to tensor [0, 255]
    transform_pil_tensor = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.PILToTensor(),
        ])

    # Transform image to tensor [-1, 1]
    transform_tensor = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize
        ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    if before== True:
        image = transform_pil_tensor(raw_image).to(device) 
    else:
        image = transform_tensor(raw_image).unsqueeze(0).to(device) 

    return image


def load_image_for_sat(id, image_size, device, before=True, dataset='flicker8k'):

    '''
    input an pil.image the function will convert it into 
    image tensor range [0, 255] or [-1. 1]
    '''
    if dataset == 'coco':
        # _, _, raw_image = get_coco_dataset_for_sat(id)
        folder_path = '/home/jiyli/Data/Image_Attack/data/train2014'
        files = os.listdir(folder_path)
        files = sorted(files)
        img = imread(folder_path + '/' + str(files[id]))
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = imresize(img, (255, 255))
        img = img.transpose(2, 0, 1)
        if not before:
            img = img / 255.
            img = torch.FloatTensor(img).to(device)
        
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([normalize])
            img = transform(img)  # (3, image_size, image_size)
        else:
            img = torch.from_numpy(img)
    if dataset == 'flicker8k':
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
            img = transform(image_resized)
        if not before:
            img = img / 255.
            img = torch.FloatTensor(img).to(device)
        
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            transform = transforms.Compose([normalize])
            
            img = transform(img)  # (3, image_size, image_size)
        # else:
        #     img = torch.from_numpy(img)

    return img

# Divide candidate to [[pixel_1], [pixel_2], [pixel_3]]

def divide_list(lst, num_chunks=5):
    '''
    divide candidates by 5 (5 elelments each candidate)
    '''
    chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks
    result = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size
        if remainder > 0:
            end += 1
            remainder -= 1
        result.append(lst[start:end])
        start = end
    return result

# Save the perturbed image
def save_image(id, pixels, figure, is_attention='att'):
    '''
    save image
    '''
    transform_toimage = transforms.ToPILImage()
    test_figure = transform_toimage(figure)
    figure_ = test_figure.save('outputs/blip_flicker8k_1000p_100samples_overall_384_attacked/%d_%d_%s.jpg' %(id, pixels, is_attention))

    return 'saved'

def save_raw_image(id):
    image_384 = dataloader.get_item('coco', id, 384)[1]
    transform_toimage = transforms.ToPILImage()
    test_figure = transform_toimage(image_384)
    figure_ = test_figure.save('outputs/demo_paper/%d_raw_image.jpg' %(id))
    return 'saved'
    
def convert_to_preferred_format(sec):
    '''
    calculalte time cumsumption
    '''
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec) 


def load_json(dir):
    '''
    load json
    '''
    file = open(dir, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def image_path(id, dataset='coco'):
    '''
    return image path
    '''
    if dataset=='coco':
        folder_path = '/home/jiyli/Data/Image_Attack/data/train2014'
        files = os.listdir(folder_path)
        files = sorted(files)
        return folder_path + '/' + str(files[id])
    if dataset == 'flicker8k':
        image_dir = '/home/jiyli/Data/Image_Attack/captionattack/Flicker8k_Dataset'
        image_filenames = sorted(os.listdir(image_dir))
        if 0 <= id < len(image_filenames):
            image_filename = image_filenames[id]
            image_path = os.path.join(image_dir, image_filename)
            return image_path

def filter_alphas():
    '''
    pick pixels from alphas
    '''
    # Convert the tensor to a numpy array
    output_array = output_tensor.numpy()

    # Define the file path
    file_path = 'alphas.txt'

    # Write the tensor to a text file
    with open(file_path, 'w') as file:
        # Iterate over each element in the array and write it to the file
        for row in output_array:
            for element in row:
                file.write(str(element) + ' ')
            file.write('\n')

    # Read the tensor data from the file
    output_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Convert the data to a tensor
        for line in lines:
            elements = line.strip().split(' ')
            row = [float(element) for element in elements]
            output_data.append(row)
    output_tensor = torch.tensor(output_data)
    file.close()
    # Print the tensor
    print(output_tensor)

    pass


def circle_area(image_path, pixel_coordinates, radius=10, outline_color=(255, 0, 0), outline_width=2):

    # convert <best_pixels> to coordinates pairs
    pixels = [(i[0], i[1]) for i in pixel_coordinates]
    # Open the image
    image = Image.open(image_path)

    # Create a new image with RGBA mode to support transparency
    circled_image = Image.new("RGBA", image.size)
    circled_image.paste(image, (0, 0))

    # Create a draw object
    draw = ImageDraw.Draw(circled_image)

    # Calculate the bounding box for the entire area defined by the pixel coordinates
    min_x = min(pixel_coordinates, key=lambda coord: coord[0])[0]
    min_y = min(pixel_coordinates, key=lambda coord: coord[1])[1]
    max_x = max(pixel_coordinates, key=lambda coord: coord[0])[0]
    max_y = max(pixel_coordinates, key=lambda coord: coord[1])[1]
    bbox = (min_x, min_y, max_x, max_y)

    # Draw the circle outline
    draw.ellipse(bbox, outline=outline_color, width=outline_width)

    return circled_image

def draw_multi_image(id):

    # Specify the paths to the individual image files
    path1 = f'outputs/separate_test_500p_lm/{id}_separate.png'
    path2 = f'outputs/separate_test_500p_lm/{id}_att_allin.png'
    path3 = f'outputs/separate_test_500p_lm/{id}_att_sep.png'
    path4 = f'outputs/separate_test_500p_lm/{id}_noatt.png'
    path5 = f'outputs/separate_test_500p_lm/{id}_allin.png'
    # Load the individual image files
    image1 = Image.open(path1)
    image2 = Image.open(path2)
    image3 = Image.open(path3)
    image4 = Image.open(path4)
    image5 = Image.open(path5)
    width = max(image1.width + image2.width, image3.width + image4.width)
    height = image1.height + image3.height + max(image2.height, image4.height) + image5.height

    # Create a new image file to combine the individual images
    combined_image = Image.new('RGB', (width, height))

    # Paste the individual images into the combined image file
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))
    combined_image.paste(image3, (0, image1.height))
    combined_image.paste(image4, (image3.width, image2.height))
    combined_image.paste(image5, ((width - image5.width) // 2, image1.height + image3.height))

    # Trim the excess white space
    combined_image = combined_image.crop(combined_image.getbbox())
    fig, ax = plt.subplots()
    ax.imshow(combined_image)
    ax.axis('off')

    # Define the labels for each image
    label1 = "separate"
    label2 = "attention_all_in"
    label3 = "attention_separate"
    label4 = "no_attention"
    label5 = 'allin'
    label_x1 = image1.width // 2
    label_x2 = image1.width + image2.width // 2
    label_x3 = image3.width // 2
    label_x4 = image3.width + image4.width // 2
    label_x5 = image5.width // 2
    label_y = height + 10

    ax.text(label_x1, label_y - image3.height, label1, ha='center')
    ax.text(label_x2, label_y - image2.height, label2, ha='center')
    ax.text(label_x3, label_y, label3, ha='center')
    ax.text(label_x4, label_y, label4, ha='center')
    ax.text(label_x5, label_y, label5, ha='center')

    # Save the combined image file
    plt.savefig(f'outputs/separate_combined_img/{id}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':
    # # dump_result = load_json('outputs/variation20_500-3000.json')
    # # print(dump_result, len(dump_result[0]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(load_image_for_sat(3, device=device, image_size=384, before=False))

    # # Example usage
    # image_path = "path/to/your/image.jpg"
    # pixel_coordinates = [(100, 200), (300, 400), (500, 600)]
    # circled_image = circle_area(image_path, pixel_coordinates)

    # # Save the circled image
    # circled_image.save("path/to/save/circled_image.jpg")
    # print(load_image_for_sat(78, image_size=255, device=device, dataset='flicker8k'))
    print(image_path(90,'flicker8k'))