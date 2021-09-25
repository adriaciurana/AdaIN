from torch.nn import functional as F
from utils import compute_mu_sigma

def content_loss(feat_g_t, feat_content):
    return F.mse_loss(feat_g_t, feat_content)


def style_loss(phi_list_g_t, phi_list_style):
    loss_style = 0.
    for phi_list_g_t_step, phi_list_style_step in zip(phi_list_g_t.values(), phi_list_style.values()):
        mu_g_t_step, sigma_g_t_step = compute_mu_sigma(phi_list_g_t_step)
        mu_style_step, sigma_style_step = compute_mu_sigma(phi_list_style_step)
        loss_style += F.mse_loss(mu_g_t_step, mu_style_step)
        loss_style += F.mse_loss(sigma_g_t_step, sigma_style_step)
    return loss_style / len(phi_list_g_t)