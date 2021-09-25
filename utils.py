import torch
from torchvision import transforms

def compute_mu_sigma(x):
    mu = torch.mean(x + 1e-6, dim=(2, 3))
    sigma = torch.std(x + 1e-6, dim=(2, 3))

    return mu, sigma

def apply_mu_sigma(x_norm, mu, sigma):
    return (x_norm * sigma[..., None, None]) + mu[..., None, None]

def norm_by_mu_sigma(x, mu, sigma):
    return (x - mu[..., None, None]) / (sigma[..., None, None] + 1e-6)

def adaIN(feat_content, feat_style):
    mu_content, sigma_content = compute_mu_sigma(feat_content)
    mu_style, sigma_style = compute_mu_sigma(feat_style)
    feat_norm_content = norm_by_mu_sigma(feat_content, mu_content, sigma_content)
    t = apply_mu_sigma(feat_norm_content, mu_style, sigma_style)
    return t

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = torch.tensor(self.mean, device=tensor.device, dtype=tensor.dtype)[None, :, None, None]
        std = torch.tensor(self.std, device=tensor.device, dtype=tensor.dtype)[None, :, None, None]
        tensor.mul_(std).add_(mean)
        return tensor

def create_mosaic(dataloader, model, device, num_batches=20):
    iter_dataloader = iter(dataloader)

    images_content_unnorm = []
    images_style_unnorm = []
    images_content_styled_unnorm = []

    with torch.no_grad():
        for _ in range(num_batches):
            try:
                images_content, images_style = next(iter_dataloader)
                images_content = images_content.to(device)
                images_style = images_style.to(device)

                g_t = model(images_content, images_style)

                images_content_unnorm.append(torch.clip(unnormalize(images_content), 0., 1.))
                images_style_unnorm.append(torch.clip(unnormalize(images_style), 0., 1.))
                images_content_styled_unnorm.append(torch.clip(unnormalize(g_t), 0., 1.))
            
            except StopIteration:
                iter_dataloader = iter(dataloader)

    return (
        torch.cat(images_content_unnorm, dim=0),
        torch.cat(images_style_unnorm, dim=0),
        torch.cat(images_content_styled_unnorm, dim=0)
    )


class NaiveWith:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

unnormalize = UnNormalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)