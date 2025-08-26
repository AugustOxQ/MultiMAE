import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

image_mean = np.array([0.5, 0.5, 0.5])
image_std = np.array([0.5, 0.5, 0.5])

def show_image(image, title=''):
    assert image.shape[2] == 3
    denormalized = image * image_std + image_mean
    img_display = torch.clip(denormalized, 0, 1).numpy() * 255
    img_display = img_display.astype(np.uint8)
    plt.imshow(img_display)
    plt.title(title, fontsize=16)
    plt.axis('off')

def run_inference(image, model):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    model.eval()
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)
    model.to("cpu")
    image = image.to("cpu")

    with torch.no_grad():
        out, mask = model(image)
        y = out.detach().cpu()
        mask = mask.detach().cpu()
        x = image.permute(0, 2, 3, 1).cpu()
        y = y.permute(0, 2, 3, 1)
        mask = mask.permute(0, 2, 3, 1)
        im_masked = x * (1 - mask)
        im_paste = x * (1 - mask) + y * mask

        plt.rcParams['figure.figsize'] = [24, 24]
        plt.subplot(1, 4, 1)
        show_image(x[0], "Original")
        plt.subplot(1, 4, 2)
        show_image(im_masked[0], "Masked")
        plt.subplot(1, 4, 3)
        show_image(y[0], "Reconstruction")
        plt.subplot(1, 4, 4)
        show_image(im_paste[0], "Reconstruction + Visible")
        plt.show()


