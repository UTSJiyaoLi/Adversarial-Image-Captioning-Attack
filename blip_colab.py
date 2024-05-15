import torch
from utils import load_image
from blip_model import model
from utils import device


def blip(image, model=model, image_size=384):
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=50, min_length=5)
    return caption[0]

if __name__ == '__main__':
    id = 3
    image = load_image(id = id, image_size=384, device=device, before=False)
    print(blip(image=image))