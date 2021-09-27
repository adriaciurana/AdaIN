import torch
from torch.nn import functional as F
from utils import compute_mu_sigma

# def content_loss(feat_g_t, feat_content):
#     w = torch.abs(feat_content) + 0.1
#     return (w * (feat_g_t - feat_content) ** 2).mean()

def content_loss(feat_g_t, feat_content):
    fc_dx = torch.diff(feat_content, dim=1)
    fc_dy = torch.diff(feat_content, dim=2)
    gt_dx = torch.diff(feat_content, dim=1)
    gt_dy = torch.diff(feat_content, dim=2)

    color_loss = 0.5 * F.mse_loss(feat_g_t, feat_content)
    diff_loss = 0.5 * (F.mse_loss(fc_dx, gt_dx) + F.mse_loss(fc_dy, gt_dy))
    return color_loss + diff_loss


def style_loss(phi_list_g_t, phi_list_style):
    loss_style = 0.
    for phi_list_g_t_step, phi_list_style_step in zip(phi_list_g_t.values(), phi_list_style.values()):
        mu_g_t_step, sigma_g_t_step = compute_mu_sigma(phi_list_g_t_step)
        mu_style_step, sigma_style_step = compute_mu_sigma(phi_list_style_step)
        loss_style += F.mse_loss(mu_g_t_step, mu_style_step)
        loss_style += F.mse_loss(sigma_g_t_step, sigma_style_step)
    return loss_style / len(phi_list_g_t)