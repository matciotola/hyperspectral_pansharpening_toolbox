import inspect
import os

import torch
from scipy import io
from skimage.transform import rescale
from tqdm import tqdm

from Utils.dl_tools import open_config
from Utils.spectral_tools import gen_mtf
from .aux import local_corr_mask, pca, inverse_pca, normalize, denormalize
from .loss import SpectralLoss, StructuralLoss
from .network import PCA_Z_PNN_model


def PCA_Z_PNN(ordered_dict):
    config_path = os.path.join(os.path.dirname(inspect.getfile(PCA_Z_PNN_model)), 'config.yaml')

    config = open_config(config_path)
    #os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda:" + config.gpu_number if torch.cuda.is_available() else "cpu")

    pan = torch.clone(ordered_dict.pan).float()
    ms_lr = torch.clone(ordered_dict.ms_lr).float()

    pan = normalize(pan, nbands=1, nbits=16)
    fused, ta_history = target_adaptation_and_prediction(device, ms_lr, pan, config, ordered_dict)

    if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(PCA_Z_PNN_model)), 'Stats', 'PCA-Z-PNN')):
        os.makedirs(os.path.join(os.path.dirname(inspect.getfile(PCA_Z_PNN_model)), 'Stats', 'PCA-Z-PNN'))
    io.savemat(os.path.join(os.path.dirname(inspect.getfile(PCA_Z_PNN_model)), 'Stats', 'PCA-Z-PNN',
                            'Target_Adaptation_PCA-Z-PNN_{}.mat'.format(ordered_dict.name)),
               ta_history)

    torch.cuda.empty_cache()
    return fused.detach().cpu().double()


def target_adaptation_and_prediction(device, ms_lr, pan, config, ordered_dict):
    pan = torch.clone(pan).to(device)
    wl = ordered_dict.wavelenghts

    num_blocks = config.num_blocks
    n_components = config.n_components

    criterion_spec = SpectralLoss(
        gen_mtf(ordered_dict.ratio, ordered_dict.dataset, kernel_size=61, nbands=n_components), ordered_dict.ratio,
        device).to(device)
    criterion_struct = StructuralLoss(ordered_dict.ratio, device).to(device)

    last_wl = config.last_wl

    history_loss_spec = []
    history_loss_struct = []

    alpha = config.alpha_1
    epochs = config.epochs

    fused = []

    band_blocks = []

    band_rgb = 0
    while wl[band_rgb] < last_wl:
        band_rgb += 1

    band_blocks.append(ms_lr[:, :band_rgb + 1, :, :])
    band_blocks.append(ms_lr[:, band_rgb:, :, :])

    # for block in band_blocks:
    for block_index in range(num_blocks):

        net = PCA_Z_PNN_model(nbands=n_components).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2))
        net.train()

        ms_lr_pca, W, mu = pca(band_blocks[block_index])

        ms_pca = torch.tensor(rescale(torch.squeeze(ms_lr_pca).numpy(), ordered_dict.ratio, order=3, channel_axis=0))[
                 None, :, :, :]
        spec_ref_exp = normalize(ms_pca[:, :n_components, :, :], nbands=ms_pca.shape[1], nbits=16).to(device)
        spec_ref = normalize(ms_lr_pca[:, :n_components, :, :], nbands=ms_pca.shape[1], nbits=16).to(device)

        min_loss = torch.inf

        inp = torch.cat([spec_ref_exp, pan], dim=1)

        # Aux data generation

        threshold = local_corr_mask(inp, ordered_dict.ratio, ordered_dict.dataset, device, config.semi_width)

        if block_index == 1:
            alpha = config.alpha_2

        print('Block index {} / {}'.format(block_index + 1, num_blocks))

        pbar = tqdm(range(epochs))

        for epoch in pbar:

            pbar.set_description('Epoch %d/%d' % (epoch + 1, epochs))

            net.train()
            optim.zero_grad()

            outputs = net(inp)

            loss_spec = criterion_spec(outputs, spec_ref)
            loss_struct, loss_struct_without_threshold = criterion_struct(outputs[:, :1, :, :], pan,
                                                                          threshold[:, :1, :, :])

            loss = loss_spec + alpha * loss_struct

            loss.backward()
            optim.step()

            running_loss_spec = loss_spec.item()
            running_loss_struct = loss_struct_without_threshold

            history_loss_spec.append(running_loss_spec)
            history_loss_struct.append(running_loss_struct)

            if loss.item() < min_loss:
                min_loss = loss.item()
                if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(PCA_Z_PNN_model)), 'temp')):
                    os.makedirs(os.path.join(os.path.dirname(inspect.getfile(PCA_Z_PNN_model)), 'temp'))
                torch.save(net.state_dict(), os.path.join(os.path.dirname(inspect.getfile(PCA_Z_PNN_model)), 'temp',
                                                          'PCA-Z-PNN_best_model.tar'))

            pbar.set_postfix(
                {'Spec Loss': running_loss_spec, 'Struct Loss': running_loss_struct})

        net.eval()
        net.load_state_dict(
            torch.load(
                os.path.join(os.path.dirname(inspect.getfile(PCA_Z_PNN_model)), 'temp', 'PCA-Z-PNN_best_model.tar')))

        ms_pca[:, :n_components, :, :] = denormalize(net(inp), nbands=ms_pca.shape[1], nbits=16)
        fused_block = inverse_pca(ms_pca, W, mu)

        if block_index == 0:
            fused.append(fused_block[:, :-1, :, :].detach().cpu())
        else:
            fused.append(fused_block.detach().cpu())

    fused = torch.cat(fused, 1)
    history = {'loss_spec': history_loss_spec, 'loss_struct': history_loss_struct}

    return fused, history
