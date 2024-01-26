import gc
import os
from scipy import io
import inspect
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from .aux import get_noise, initialize_weights, TrainingDatasetKite, TestDatasetKite
from .network import DIP, PanPredictionNetwork, Downsampler, KiteNetwork
from Utils.dl_tools import open_config, generate_paths, TrainingDatasetRR, normalize, denormalize


def DIP_HyperKite(ordered_dict):
    config_path = os.path.join(os.path.dirname(inspect.getfile(KiteNetwork)), 'config.yaml')

    config = open_config(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pan = torch.clone(ordered_dict.pan).float()
    ms_lr = torch.clone(ordered_dict.ms_lr).float()

    model_weights_path = config.model_weights_path

    net = KiteNetwork(ms_lr.shape[1], ordered_dict.ratio)

    if not config.train or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join(os.path.dirname(inspect.getfile(KiteNetwork)), 'weights',
                                              ordered_dict.dataset + '.tar')
        if os.path.exists(model_weights_path):
            net.load_state_dict(torch.load(model_weights_path))
    else:
        initialize_weights(net)

    net = net.to(device)

    if config.train:
        if config.training_img_root == '':
            training_img_root = ordered_dict.root
        else:
            training_img_root = config.training_img_root
        train_paths = generate_paths(training_img_root, ordered_dict.dataset, 'Training')
        ds_train = TrainingDatasetRR(train_paths, normalize)
        train_loader = DataLoader(ds_train, batch_size=1, shuffle=False)
        prior_images = prior_execution(device, train_loader, config)

        ds_train = TrainingDatasetKite(train_paths, prior_images, normalize)
        train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)

        if config.validation:
            val_paths = generate_paths(training_img_root, ordered_dict.dataset, 'Validation')
            ds_val = TrainingDatasetRR(val_paths, normalize)
            val_loader = DataLoader(ds_val, batch_size=1, shuffle=False)

            prior_images_val = prior_execution(device, val_loader, config)

            ds_val = TrainingDatasetKite(val_paths, prior_images_val, normalize)
            val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)

        else:
            val_loader = None

        history = train_kite(device, net, train_loader, config, val_loader)

        if config.save_weights:
            if not os.path.exists(
                    os.path.join(os.path.dirname(inspect.getfile(KiteNetwork)), config.save_weights_path)):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(KiteNetwork)), config.save_weights_path))
            torch.save(net.state_dict(),
                       os.path.join(os.path.dirname(inspect.getfile(KiteNetwork)), config.save_weights_path,
                                    ordered_dict.dataset +'.tar'))

        if config.save_training_stats:
            if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(KiteNetwork)), 'Stats', 'HyperKite')):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(KiteNetwork)), 'Stats', 'HyperKite'))
            io.savemat(
                os.path.join(os.path.dirname(inspect.getfile(KiteNetwork)), 'Stats', 'HyperKite', 'Training_HyperKite_' + ordered_dict.dataset + ' .mat'),
                history)

    pan = normalize(pan)
    ms_lr = normalize(ms_lr)

    pan = pan.to(device)
    ms_lr = ms_lr.to(device)
    net.eval()

    test_ds = TestDatasetKite(pan, ms_lr)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    prior = prior_execution(device, test_loader, config)

    del test_ds, test_loader, ms_lr
    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        outputs_patches = []
        kc, kh, kw = prior.shape[1] + 1, 180, 180  # kernel size
        dc, dh, dw = prior.shape[1] + 1, 180, 180  # stride
        patches = torch.cat([prior.cpu(), pan.cpu()], 1).unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = list(patches.shape)
        patches = patches.contiguous().view(-1, kc, kh, kw).to(device)

        patches = torch.nn.functional.pad(patches, (6, 6, 6, 6), mode='reflect')

        prior_patches = patches[:, :-1, :, :]
        pan_patches = patches[:, -1, :, :].unsqueeze(1)

        for i in range(prior_patches.shape[0]):
            patch = net(pan_patches[i:i+1, :, :, :].to(device), prior_patches[i:i+1, :, :, :].to(device))
            outputs_patches.append(patch.detach().cpu())

        # Image reconstruction
        outputs_patches = torch.cat(outputs_patches, 0)
        outputs_patches = outputs_patches[:, :, 6:-6, 6:-6]
        unfold_shape[4] = unfold_shape[4] - 1
        fused = outputs_patches.view(unfold_shape)
        output_c = unfold_shape[1] * unfold_shape[4]
        output_h = unfold_shape[2] * unfold_shape[5]
        output_w = unfold_shape[3] * unfold_shape[6]
        fused = fused.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        fused = fused.view(1, output_c, output_h, output_w)

    fused = denormalize(torch.clip(fused,0, 1))
    return fused.detach().cpu().double()


def prior_execution(device, train_loader, config):
    prior_hs = []

    criterion_spectral = nn.L1Loss().to(device)
    criterion_spatial = nn.L1Loss().to(device)

    pbar = tqdm(train_loader)

    for i, data in enumerate(pbar):
        pbar.set_description('Applying Prior: %d/%d' % (i + 1, len(train_loader)))
        pan, ms_lr, _, _ = data
        pan = pan.to(device)
        ms_lr = ms_lr.to(device)
        spectral_bands_number = ms_lr.shape[1]

        net_input = get_noise(config.input_depth, (pan.shape[3], pan.shape[2])).type(pan.dtype)

        net = DIP(config.input_depth, spectral_bands_number).to(device)
        pan_network = PanPredictionNetwork(spectral_bands_number).to(device)
        downsampler = Downsampler(spectral_bands_number, config.factor, config.kernel_type, config.phase,
                                  preserve_size=True).to(device)

        optim = torch.optim.Adam(
            list(net.parameters()) + list(pan_network.parameters()),
            lr=config.dhp_learning_rate,
            betas=(config.beta_1, config.beta_2)
            )
        scheduler = torch.optim.lr_scheduler.StepLR(optim, config.step_size, config.gamma)

        net.train()
        pan_network.train()
        # downsampler.train()

        for epoch in range(config.epochs):

            optim.zero_grad()

            # Closure

            outputs_hr = net(net_input.to(device))
            predicted_pan = pan_network(outputs_hr)
            outputs_lr = downsampler(outputs_hr)
            loss_spectral = criterion_spectral(outputs_lr, ms_lr)
            loss_spatial = criterion_spatial(predicted_pan, pan)

            loss = loss_spectral + config.alpha * loss_spatial

            pbar.set_postfix({'Spec Loss': loss_spectral.item(), 'Spa Loss': loss_spatial.item()})

            loss.backward()
            optim.step()
            scheduler.step()

        final_outputs_hr = net(net_input.to(device)).detach().cpu()
        prior_hs.append(final_outputs_hr)
    prior_hs = torch.cat(prior_hs, 0)
    prior_hs = torch.clip(prior_hs, 0, 1)

    del final_outputs_hr, net_input, net, pan_network, downsampler
    gc.collect()
    torch.cuda.empty_cache()


    return prior_hs


def train_kite(device, net, train_loader, config, val_loader=None):
    criterion = nn.L1Loss().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=config.kite_learning_rate, betas=(config.beta_1, config.beta_2),
                             weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                          step_size=config.kite_step_size,
                                          gamma=config.kite_gamma)

    history_loss = []
    history_val_loss = []

    pbar = tqdm(range(config.kite_epochs))

    for epoch in pbar:
        pbar.set_description('Epoch %d/%d' % (epoch + 1, config.kite_epochs))
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
            scheduler.step()

            running_loss += loss.item()

        running_loss = running_loss / len(train_loader)

        if val_loader is not None:
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    pan, _, priors, gt = data
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
