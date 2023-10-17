import os
from scipy import io
import torch
import inspect
from torch.utils.data import DataLoader
from tqdm import tqdm

from .network import R_PNN_model
from .loss import SpectralLoss, StructuralLoss
from .aux import local_corr_mask

from Utils.dl_tools import open_config, generate_paths, TrainingDatasetFR, normalize, denormalize
from Utils.spectral_tools import gen_mtf


def R_PNN(ordered_dict):

    config_path = os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), 'config.yaml')

    config = open_config(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pan = torch.clone(ordered_dict.pan).float()
    ms = torch.clone(ordered_dict.ms).float()
    ms_lr = torch.clone(ordered_dict.ms_lr)

    model_weights_path = config.model_weights_path

    net = R_PNN_model()

    if not config.train or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join(os.getcwd(), 'weights', 'R-PNN.tar')
        if os.path.exists(model_weights_path):
            net.load_state_dict(torch.load(model_weights_path))

    net = net.to(device)

    if config.train:
        if config.training_img_root == '':
            training_img_root = ordered_dict.root
        else:
            training_img_root = config.training_img_root
        train_paths = generate_paths(training_img_root, ordered_dict.dataset, 'Training')
        ds_train = TrainingDatasetFR(train_paths, normalize)
        train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)

        if config.validation:
            val_paths = generate_paths(training_img_root,  ordered_dict.dataset, 'Validation')
            ds_val = TrainingDatasetFR(val_paths, normalize)
            val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)
        else:
            val_loader = None

        history = train(device, net, train_loader, config, ordered_dict, val_loader)

        if config.save_weights:
            if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), config.save_weights_path)):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), config.save_weights_path))
            torch.save(net.state_dict(), os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), config.save_weights_path, 'R-PNN.tar'))

        if config.save_training_stats:
            if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), 'Stats', 'R-PNN')):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), 'Stats', 'R-PNN'))
            io.savemat(os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), 'Stats', 'R-PNN', 'Training_R-PNN.mat'), history)

    pan = normalize(pan)
    ms = normalize(ms)
    ms_lr = normalize(ms_lr)

    net.eval()
    fused, ta_history = target_adaptation_and_prediction(device, net, ms_lr, ms, pan, config, ordered_dict)

    if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), 'Stats', 'R-PNN')):
        os.makedirs(os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), 'Stats', 'R-PNN'))
    io.savemat(os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), 'Stats', 'R-PNN', 'Target_Adaptation_R-PNN_{}.mat'.format(ordered_dict.name)),
               ta_history)

    fused = denormalize(fused)
    torch.cuda.empty_cache()
    return fused.detach().cpu().double()


def train(device, net, train_loader, config, ordered_dict, val_loader=None):

    criterion_spec = SpectralLoss(gen_mtf(ordered_dict.ratio, ordered_dict.dataset, kernel_size=61, nbands=1), ordered_dict.ratio, device).to(device)
    criterion_struct = StructuralLoss(ordered_dict.ratio, device)
    optim = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    history_loss_spec = []
    history_loss_struct = []
    history_val_loss_spec = []
    history_val_loss_struct = []

    pbar = tqdm(range(config.epochs))

    for epoch in pbar:

        pbar.set_description('Epoch %d/%d' % (epoch + 1, config.epochs))
        running_loss_spec = 0.0
        running_loss_struct = 0.0
        running_val_loss_spec = 0.0
        running_val_loss_struct = 0.0

        net.train()

        for i, data in enumerate(train_loader):
            optim.zero_grad()

            pan, ms_lr, ms = data
            pan_band = pan.to(device)
            band = ms[:, 0:1, :, :].to(device)
            band_lr = ms_lr[:, 0:1, :, :].to(device)

            # Aux data generation

            inp = torch.cat([band, pan_band], dim=1)
            threshold = local_corr_mask(inp, ordered_dict.ratio, ordered_dict.dataset, device, config.semi_width)

            outputs = net(inp)

            loss_spec = criterion_spec(outputs, band_lr)
            loss_struct, loss_struct_without_threshold = criterion_struct(outputs, pan_band, threshold)

            loss = loss_spec + config.alpha_1 * loss_struct

            loss.backward()
            optim.step()

            running_loss_spec += loss_spec.item()
            running_loss_struct += loss_struct_without_threshold

        running_loss_spec = running_loss_spec / len(train_loader)
        running_loss_struct = running_loss_struct / len(train_loader)


        if val_loader is not None:
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    pan, ms_lr, ms = data
                    pan = pan.to(device)
                    ms = ms[:, 0:1, :, :].to(device)
                    ms_lr = ms_lr[:, 0:1, :, :].to(device)

                    inp = torch.cat([ms, pan], dim=1)
                    threshold = local_corr_mask(inp, ordered_dict.ratio, ordered_dict.dataset, device, config.semi_width)

                    outputs = net(inp)

                    val_loss_spec = criterion_spec(outputs, ms_lr)
                    _, val_loss_struct_without_threshold = criterion_struct(outputs, pan, threshold)

                    running_val_loss_spec += val_loss_spec.item()
                    running_val_loss_struct += val_loss_struct_without_threshold

            running_val_loss_spec = running_val_loss_spec / len(val_loader)
            running_val_loss_struct = running_val_loss_struct / len(val_loader)

        history_loss_spec.append(running_loss_spec)
        history_loss_struct.append(running_loss_struct)
        history_val_loss_spec.append(running_val_loss_spec)
        history_val_loss_struct.append(running_val_loss_struct)


        pbar.set_postfix(
            {'Spec Loss': running_loss_spec, 'Struct Loss': running_loss_struct, 'Val Spec Loss': running_val_loss_spec, 'Val Struct Loss': running_val_loss_struct})

    history = {'loss_spec': history_loss_spec, 'loss_struct': history_loss_struct, 'val_loss_spec': history_val_loss_spec, 'val_loss_struct': history_val_loss_struct}

    return history


def target_adaptation_and_prediction(device, net, ms_lr, ms, pan, config, ordered_dict):

    ms = torch.clone(ms)
    pan = torch.clone(pan).to(device)
    wl = ordered_dict.wavelenghts

    criterion_spec = SpectralLoss(gen_mtf(ordered_dict.ratio, ordered_dict.dataset, kernel_size=61, nbands=1), ordered_dict.ratio, device).to(device)
    criterion_struct = StructuralLoss(ordered_dict.ratio, device)
    optim = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

    history_loss_spec = []
    history_loss_struct = []

    alpha = config.alpha_1

    fused = []

    for band_number in range(ms.shape[1]):

        band = ms[:, band_number:band_number + 1, :, :].to(device)
        band_lr = ms_lr[:, band_number:band_number + 1, :, :].to(device)

        # Aux data generation
        inp = torch.cat([band, pan], dim=1)
        threshold = local_corr_mask(inp, ordered_dict.ratio, ordered_dict.dataset, device, config.semi_width)

        if wl[band_number] > 700:
            alpha = config.alpha_2

        if band_number == 0:
            ft_epochs = config.first_iter
        else:
            ft_epochs = int(min(((wl[band_number] - wl[band_number - 1]) // 10 + 1) * config.epoch_nm, config.sat_val))
        min_loss = torch.inf
        print('Band {} / {}'.format(band_number + 1, ft_epochs))
        pbar = tqdm(range(ft_epochs))

        for epoch in pbar:

            pbar.set_description('Epoch %d/%d' % (epoch + 1, ft_epochs))

            net.train()
            optim.zero_grad()

            outputs = net(inp)

            loss_spec = criterion_spec(outputs, band_lr)
            loss_struct, loss_struct_without_threshold = criterion_struct(outputs, pan, threshold)

            loss = loss_spec + alpha * loss_struct

            loss.backward()
            optim.step()

            running_loss_spec = loss_spec.item()
            running_loss_struct = loss_struct_without_threshold

            history_loss_spec.append(running_loss_spec)
            history_loss_struct.append(running_loss_struct)

            if loss.item() < min_loss:
                min_loss = loss.item()
                if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), 'temp')):
                    os.makedirs(os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), 'temp'))
                torch.save(net.state_dict(), os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), 'temp', 'R-PNN_best_model.tar'))

            pbar.set_postfix(
                {'Spec Loss': running_loss_spec, 'Struct Loss': running_loss_struct})
        net.load_state_dict(torch.load(os.path.join(os.path.dirname(inspect.getfile(R_PNN_model)), 'temp', 'R-PNN_best_model.tar')))
        fused.append(net(inp).detach().cpu())

    fused = torch.cat(fused, 1)
    history = {'loss_spec': history_loss_spec, 'loss_struct': history_loss_struct}

    return fused, history