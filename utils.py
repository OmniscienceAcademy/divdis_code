import torch
import torchvision
import numpy as np
import random
from defaults import *
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F


def set_seed(seed=0):
    print('Seeding pseudo-random functions with seed {}...'.format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def show(imgs):
    """
    Plots list of images
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img).astype(int))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def show_batch(batch: torch.Tensor):
    """
    Shows 4D batch of tensors
    """
    grid = torchvision.utils.make_grid(batch)
    show(grid)


def test_containers_and_models(container, models, device):
    target_test = container.target(train=False)
    target_loader = DataLoader(target_test, 32, shuffle=True)
    target_batch, target_labels, target_alt_labels = iter(target_loader).next()

    current_batch = target_batch[0:6].to(device)

    outs = [model(current_batch) for model in models]

    for out in outs:
        print(torch.round(out, decimals=2))

    show_batch(current_batch)
    print(target_labels[0:6])
    print(target_alt_labels[0:6])


def save_params(models, name):
    for i, model in enumerate(models):
        ppath = os.path.join(PARAM_PATH, f"{name}_{i}")
        print('Saving parameters: {}'.format(ppath))
        torch.save(model.state_dict(), ppath)


def load_params(models, name):
    for i, model in enumerate(models):
        ppath = os.path.join(PARAM_PATH, f"{name}_{i}")
        print('Loading parameters: {}'.format(ppath))
        model.load_state_dict(torch.load(ppath))


def add_text(image, text):
    my_image = Image.open(image)
    image_editable = ImageDraw.Draw(my_image)
    myFont = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 50)
    W, H = my_image.size
    w, h = image_editable.textsize(text, font=myFont)
    image_editable.text(((W - w) / 2, ((H) * 8 / 9) - h / 2), text, fill="red", font=myFont)
    return my_image
