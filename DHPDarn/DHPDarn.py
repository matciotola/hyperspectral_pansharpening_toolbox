import gc
import os
from scipy import io
import inspect
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from Utils.spectral_tools import gen_mtf, mtf_kernel_to_torch

from .aux import TrainingDatasetDarn, TestDatasetDHP
from .network import DHP, DARN, Downsampler
from Utils.dl_tools import open_config, generate_paths, TrainingDatasetRR, normalize, denormalize


def DHP_Darn(ordered_dict):

    config_path = os.path.join(os.path.dirname(inspect.getfile(DHP)), 'config.yaml')

    config = open_config(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pan = torch.clone(ordered_dict.pan).float()
    ms_lr = torch.clone(ordered_dict.ms_lr).float()

    model_weights_path = config.model_weights_path

    net = DARN(ms_lr.shape[1])

    if not (config.train and ordered_dict.img_number == 0) or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join(os.path.dirname(inspect.getfile(DHP)), 'weights', ordered_dict.dataset + '.tar')
        if os.path.exists(model_weights_path):
            net.load_state_dict(torch.load(model_weights_path))
            print('Weights loaded from: ' + model_weights_path)

    net = net.to(device)

    if (config.train or config.resume) and ordered_dict.img_number == 0:
        if config.training_img_root == '':
            training_img_root = ordered_dict.root
        else:
            training_img_root = config.training_img_root
        train_paths = generate_paths(training_img_root, ordered_dict.dataset, 'Training', 'Reduced_Resolution')

        prior_loaded_flag = False
        if config.load_priors:
            if os.path.exists(os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Priors',
                                           ordered_dict.dataset + '.tar')):
                prior_images = torch.load(os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Priors',
                                                       ordered_dict.dataset + '.tar'))
                if prior_images.shape[0] == len(train_paths):
                    print('Priors loaded')
                    prior_loaded_flag = True
        if not prior_loaded_flag:
            ds_train = TrainingDatasetRR(train_paths, normalize)
            train_loader = DataLoader(ds_train, batch_size=1, shuffle=False)
            prior_images = prior_execution(device, train_loader, config, ordered_dict)

            if config.save_priors:
                if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Priors')):
                    os.makedirs(os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Priors'))
                torch.save(prior_images, os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Priors',
                                                      ordered_dict.dataset + '.tar'))

        ds_train = TrainingDatasetDarn(train_paths, prior_images, normalize)
        train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)

        if config.validation:
            val_prior_loaded_flag = False
            if config.load_priors:
                val_paths = generate_paths(training_img_root, ordered_dict.dataset, 'Validation', 'Reduced_Resolution')
                if os.path.exists(os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Priors',
                                               'Val_' + ordered_dict.dataset + '.tar')):
                    val_prior_images = torch.load(os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Priors',
                                                               'Val_' + ordered_dict.dataset + '.tar'))
                    if val_prior_images.shape[0] == len(val_paths):
                        print('Priors loaded')
                        val_prior_loaded_flag = True
            if not val_prior_loaded_flag:
                ds_val = TrainingDatasetRR(val_paths, normalize)
                val_loader = DataLoader(ds_val, batch_size=1, shuffle=False)
                val_prior_images = prior_execution(device, val_loader, config, ordered_dict)

                if config.save_priors:
                    if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Priors')):
                        os.makedirs(os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Priors'))
                    torch.save(val_prior_images, os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Priors',
                                                              'Val_' + ordered_dict.dataset + '.tar'))

            ds_val = TrainingDatasetDarn(val_paths, val_prior_images, normalize)
            val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)

        else:
            val_loader = None

        history = train_darn(device, net, train_loader, config, val_loader)

        if config.save_weights:
            if not os.path.exists(
                    os.path.join(os.path.dirname(inspect.getfile(DHP)), config.save_weights_path)):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(DHP)), config.save_weights_path))
            torch.save(net.state_dict(),
                       os.path.join(os.path.dirname(inspect.getfile(DHP)), config.save_weights_path,
                                    ordered_dict.dataset +'.tar'))

        if config.save_training_stats:
            if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Stats', 'DARN')):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Stats', 'DARN'))
            io.savemat(
                os.path.join(os.path.dirname(inspect.getfile(DHP)), 'Stats', 'DARN', 'Training_DARN.mat'),
                history)

    pan = normalize(pan)
    ms_lr = normalize(ms_lr)


    net.eval()
    kc, kh, kw = ms_lr.shape[1], 100, 100  # kernel size
    dc, dh, dw = ms_lr.shape[1], 100, 100  # stride
    ms_lr_patches = ms_lr.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    pan_patches = pan.unfold(1, 1, 1).unfold(2, kh * ordered_dict.ratio, dh * ordered_dict.ratio).unfold(3,
                                                                                                         kw * ordered_dict.ratio,
                                                                                                         dw * ordered_dict.ratio)
    unfold_shape = list(pan_patches.shape)
    unfold_shape_ms_lr = list(ms_lr_patches.shape)
    unfold_shape[4] = unfold_shape_ms_lr[4]

    ms_lr_patches = ms_lr_patches.contiguous().view(-1, kc, kh, kw).to(device)
    pan_patches = pan_patches.contiguous().view(-1, 1, kh * ordered_dict.ratio, kw * ordered_dict.ratio).to(device)

    test_ds = TestDatasetDHP(pan_patches, ms_lr_patches)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    prior_patches = prior_execution(device, test_loader, config, ordered_dict)

    unfold_shape[4] = unfold_shape[4]
    prior = prior_patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    prior = prior.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    prior = prior.view(1, output_c, output_h, output_w)

    del test_ds, test_loader, ms_lr
    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        fused = net(pan.to(device), prior.to(device))

    fused = denormalize(torch.clip(fused, 0, 1))
    return fused.detach().cpu().double()


def prior_execution(device, train_loader, config, ordered_dict):
    prior_hs = []

    criterion_spectral = nn.L1Loss().to(device)

    pbar = tqdm(train_loader)

    for i, data in enumerate(pbar):
        pbar.set_description('Applying Prior: %d/%d' % (i + 1, len(train_loader)))
        pan, ms_lr, _, _ = data
        ms_lr = ms_lr.to(device)
        net_input = torch.zeros((pan.shape[0], ms_lr.shape[1], pan.shape[2], pan.shape[3])).float().uniform_() * 0.1
        net_input = net_input.to(device)

        net = DHP(ms_lr.shape[1]).to(device)
        downsampler = Downsampler(mtf_kernel_to_torch(gen_mtf(ordered_dict.ratio, ordered_dict.sensor, 9, ms_lr.shape[1])), ordered_dict.ratio).to(device)

        optim = torch.optim.Adam(
            net.parameters(),
            lr=config.dhp_learning_rate,
            betas=(config.beta_1, config.beta_2)
        )

        net.train()

        for epoch in range(config.epochs):

            optim.zero_grad()

            # Closure

            outputs_hr = net(net_input)
            outputs_lr = downsampler(outputs_hr)
            loss = criterion_spectral(outputs_lr, ms_lr)

            pbar.set_postfix({'Loss': loss.item()})

            loss.backward()
            optim.step()

        final_outputs_hr = net(net_input.to(device)).detach().cpu()
        prior_hs.append(final_outputs_hr)
    prior_hs = torch.cat(prior_hs, 0)
    prior_hs = torch.clip(prior_hs, 0, 1)

    del final_outputs_hr, net_input, net, downsampler
    gc.collect()
    torch.cuda.empty_cache()

    return prior_hs


def train_darn(device, net, train_loader, config, val_loader=None):
    criterion = nn.L1Loss().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=config.darn_learning_rate, betas=(config.beta_1, config.beta_2))

    history_loss = []
    history_val_loss = []

    pbar = tqdm(range(config.darn_epochs))

    for epoch in pbar:
        pbar.set_description('Epoch %d/%d' % (epoch + 1, config.darn_epochs))
        running_loss = 0.0
        running_val_loss = 0.0

        net.train()

        for i, data in enumerate(train_loader):
            optim.zero_grad()

            pan, priors, gt = data
            pan = pan.to(device)
            priors = priors.to(device)
            gt = gt.to(device)

            outputs = net(pan, priors)

            loss = criterion(outputs, gt)

            loss.backward()
            optim.step()

            running_loss += loss.item()

        running_loss = running_loss / len(train_loader)

        if val_loader is not None:
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    pan, priors, gt = data
                    pan = pan.to(device)
                    priors = priors.to(device)
                    gt = gt.to(device)

                    outputs = net(pan, priors)

                    val_loss = criterion(outputs, gt)

                    running_val_loss += val_loss.item()

            running_val_loss = running_val_loss / len(val_loader)

        history_loss.append(running_loss)
        history_val_loss.append(running_val_loss)

        pbar.set_postfix({'Loss': running_loss, 'Val Loss': running_val_loss})

    history = {'loss': history_loss, 'val_loss': history_val_loss}

    return history
