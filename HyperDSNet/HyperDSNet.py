import os
from scipy import io
import inspect
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .network import Hyper_DSNet

from Utils.dl_tools import open_config, generate_paths, TrainingDatasetRR, normalize, denormalize


def HyperDSNet(ordered_dict):

    config_path = os.path.join(os.path.dirname(inspect.getfile(Hyper_DSNet)), 'config.yaml')

    config = open_config(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pan = torch.clone(ordered_dict.pan).float()
    ms = torch.clone(ordered_dict.ms).float()
    ms_lr = torch.clone(ordered_dict.ms_lr).float()

    model_weights_path = config.model_weights_path

    net = Hyper_DSNet(ms.shape[1])

    if not config.train or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join(os.path.dirname(inspect.getfile(Hyper_DSNet)), 'weights', ordered_dict.dataset + '.tar')
        if os.path.exists(model_weights_path):
            net.load_state_dict(torch.load(model_weights_path))

    net = net.to(device)

    if config.train:
        if config.training_img_root == '':
            training_img_root = ordered_dict.root
        else:
            training_img_root = config.training_img_root
        train_paths = generate_paths(training_img_root, ordered_dict.dataset, 'Training')
        ds_train = TrainingDatasetRR(train_paths, normalize)
        train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True)

        if config.validation:
            val_paths = generate_paths(training_img_root,  ordered_dict.dataset, 'Validation')
            ds_val = TrainingDatasetRR(val_paths, normalize)
            val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)
        else:
            val_loader = None

        history = train(device, net, train_loader, config, val_loader)

        if config.save_weights:
            if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(Hyper_DSNet)), config.save_weights_path)):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(Hyper_DSNet)), config.save_weights_path))
            torch.save(net.state_dict(), os.path.join(os.path.dirname(inspect.getfile(Hyper_DSNet)), config.save_weights_path, ordered_dict.dataset + '.tar'))

        if config.save_training_stats:
            if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(Hyper_DSNet)), 'Stats', 'Hyper_DSNet')):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(Hyper_DSNet)), 'Stats', 'Hyper_DSNet'))
            io.savemat(os.path.join(os.path.dirname(inspect.getfile(Hyper_DSNet)), 'Stats', 'Hyper_DSNet', 'Training_Hyper_DSNet_' + ordered_dict.dataset +'.mat'), history)

    pan = normalize(pan)
    ms = normalize(ms)
    ms_lr = normalize(ms_lr)

    pan = pan.to(device)
    ms = ms.to(device)
    ms_lr = ms_lr.to(device)

    net.eval()
    with torch.no_grad():
        fused = net(pan, ms, ms_lr)

    fused = denormalize(fused)
    torch.cuda.empty_cache()
    return fused.detach().cpu().double()


def train(device, net, train_loader, config, val_loader=None):

    criterion = nn.MSELoss(size_average=True).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2), weight_decay=config.weight_decay)

    history_loss = []
    history_val_loss = []

    pbar = tqdm(range(config.epochs))

    for epoch in pbar:

        pbar.set_description('Epoch %d/%d' % (epoch + 1, config.epochs))
        running_loss = 0.0
        running_val_loss = 0.0

        net.train()

        for i, data in enumerate(train_loader):
            optim.zero_grad()

            pan, ms_lr, ms, gt = data
            pan = pan.to(device)
            ms_lr = ms_lr.to(device)
            ms = ms.to(device)
            gt = gt.to(device)

            outputs = net(pan, ms, ms_lr)

            loss = criterion(outputs, gt)
            loss.backward()
            optim.step()

            running_loss += loss.item()

        running_loss = running_loss / len(train_loader)


        if val_loader is not None:
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    pan, _, ms, gt = data
                    pan = pan.to(device)
                    ms = ms.to(device)
                    gt = gt.to(device)

                    outputs = net(pan, ms)

                    val_loss = criterion(outputs, gt)


                    running_val_loss += val_loss.item()

            running_val_loss = running_val_loss / len(val_loader)

        history_loss.append(running_loss)
        history_val_loss.append(running_val_loss)


        pbar.set_postfix(
            {'Loss': running_loss, 'Val Loss': running_val_loss})

    history = {'loss': history_loss, 'val_loss': history_val_loss}

    return history
