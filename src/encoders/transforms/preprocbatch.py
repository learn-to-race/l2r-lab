from torchvision import transforms
import torch

def crop_resize_center(imgs):
    assert imgs[0].shape == (
        384,
        512,
        3,
    ), f"img shape should be (384, 512, 3), found {imgs[0].shape}"
    imgs = imgs.permute(0, 3, 1, 2)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(144),
        transforms.CenterCrop(42),
        transforms.ToTensor(),
    ])
    x = torch.stack([transform(img)[:, 68:110, :] for img in imgs])
    return x