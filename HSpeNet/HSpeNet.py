import os
from scipy import io
import inspect
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .network import HSpeNet_model
from .loss import SAMLoss

from Utils.dl_tools import open_config, generate_paths, TrainingDatasetRR, normalize, denormalize


def HSpeNet(ordered_dict):

    config_path = os.path.join(os.path.dirname(inspect.getfile(HSpeNet_model)), 'config.yaml')


    config = open_config(config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pan = torch.clone(ordered_dict.pan).float()
    ms = torch.clone(ordered_dict.ms).float()

    model_weights_path = config.model_weights_path

    net = HSpeNet_model(ms.shape[1])

    if not config.train or config.resume:
        if not model_weights_path:
            model_weights_path = os.path.join(os.path.dirname(inspect.getfile(HSpeNet_model)), 'weights', 'HSpeNet.tar')
        if os.path.exists(model_weights_path):
            net.load_state_dict(torch.load(model_weights_path))

    net = net.to(device)

    if config.train:
        if config.training_img_root == '':
            training_img_root = ordered_dict.root
        else:
            training_img_root = config.training_img_root
        train_paths = generate_paths(training_img_root,  ordered_dict.dataset, 'Training')
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
            if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(HSpeNet_model)), config.save_weights_path)):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(HSpeNet_model)), config.save_weights_path))
            torch.save(net.state_dict(), os.path.join(os.path.dirname(inspect.getfile(HSpeNet_model)), config.save_weights_path, 'HSpeNet.tar'))

        if config.save_training_stats:
            if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(HSpeNet_model)), 'Stats', 'HSpeNet')):
                os.makedirs(os.path.join(os.path.dirname(inspect.getfile(HSpeNet_model)), 'Stats', 'HSpeNet'))
            io.savemat(os.path.join(os.path.dirname(inspect.getfile(HSpeNet_model)), 'Stats', 'HSpeNet', 'Training_HSpeNet.mat'), history)

    pan = normalize(pan)
    ms = normalize(ms)

    pan = pan.to(device)
    ms = ms.to(device)
    net.eval()
    with torch.no_grad():
        fused = net(pan, ms)

    fused = denormalize(fused)
    torch.cuda.empty_cache()
    return fused.detach().cpu().double()


def train(device, net, train_loader, config, val_loader=None):

    criterion_mse = nn.MSELoss().to(device)
    criterion_sam = SAMLoss().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2))

    history_loss_mse = []
    history_loss_sam = []
    history_val_loss_mse = []
    history_val_loss_sam = []

    pbar = tqdm(range(config.epochs))

    for epoch in pbar:

        pbar.set_description('Epoch %d/%d' % (epoch + 1, config.epochs))
        running_loss_mse = 0.0
        running_loss_sam = 0.0
        running_val_loss_mse = 0.0
        running_val_loss_sam = 0.0

        net.train()

        for i, data in enumerate(train_loader):
            optim.zero_grad()

            pan, _, ms, gt = data
            pan = pan.to(device)
            ms = ms.to(device)
            gt = gt.to(device)

            outputs = net(pan, ms)

            loss_mse = criterion_mse(outputs, gt)
            loss_sam = criterion_sam(outputs, gt)

            loss = config.lambda_mse * loss_mse + (1 - config.lambda_mse) * loss_sam

            loss.backward()
            optim.step()

            running_loss_mse += loss_mse.item()
            running_loss_sam += loss_sam.item()

        running_loss_mse = running_loss_mse / len(train_loader)
        running_loss_sam = running_loss_sam / len(train_loader)


        if val_loader is not None:
            net.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    pan, _, ms, gt = data
                    pan = pan.to(device)
                    ms = ms.to(device)
                    gt = gt.to(device)

                    outputs = net(pan, ms)

                    val_loss_mse = criterion_mse(outputs, gt)
                    val_loss_sam = criterion_sam(outputs, gt)

                    running_val_loss_mse += val_loss_mse.item()
                    running_val_loss_sam += val_loss_sam.item()

            running_val_loss_mse = running_val_loss_mse / len(val_loader)
            running_val_loss_sam = running_val_loss_sam / len(val_loader)

        history_loss_mse.append(running_loss_mse)
        history_loss_sam.append(running_loss_sam)
        history_val_loss_mse.append(running_val_loss_mse)
        history_val_loss_sam.append(running_val_loss_sam)


        pbar.set_postfix(
            {'MSE Loss': running_loss_mse, 'SAM Loss': running_loss_sam, 'Val MSE Loss': running_val_loss_mse, 'Val SAM Loss': running_val_loss_sam})

    history = {'loss_mse': history_loss_mse, 'loss_sam': history_loss_sam, 'val_loss_mse': history_val_loss_mse, 'val_loss_sam': history_val_loss_sam}

    return history
