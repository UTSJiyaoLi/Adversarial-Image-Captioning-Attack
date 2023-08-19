from coco import *
from utils import *


class dataloader():
    def __init__(self, name):
        self.name = name

    def get_item(name, id, image_size):
        if name == 'coco':
            img = load_image(id, image_size, device, before=False) # [-1, 1]
            new_img = load_image(id, image_size, device)
        if name == 'flicker8k':
            img = load_image(id, image_size, device, before=False,dataset=name) # [-1, 1]
            new_img = load_image(id, image_size, device,dataset=name)
        return img, new_img

    def get_item_sat(name, id, image_size):
        if name == 'coco':
            img = load_image_for_sat(id, image_size, device, before=False) # [-1, 1]
            new_img = load_image_for_sat(id, image_size, device)
        if name == 'flicker8k':
            img = load_image_for_sat(id, image_size, device, before=False,dataset=name) # [-1, 1]
            new_img = load_image_for_sat(id, image_size, device, dataset=name)
        return img, new_img

if __name__ == '__main__':
    print(dataloader.get_item_sat('coco',88,255)[1])
    