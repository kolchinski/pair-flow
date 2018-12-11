"""Train Real NVP on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util

from models import RealNVP, RealNVPLoss, PairedNVP
from tqdm import tqdm

def alternate(*args):
    for iterable in zip(*args):
        for item in iterable:
            if item is not None:
                yield item

def main(args):
    device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
    start_epoch = 0

    # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    # init
    trainloader_x, trainloader_x2, testloader_x, testloader_x2, testloader, trainloader \
        = None, None, None, None, None, None

    if args.model == 'realnvp':
        if args.dataset == 'MNIST':
            trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_test)

        elif args.dataset == 'SVHN':
            trainset = torchvision.datasets.SVHN(root='data', download=True, transform=transform_train)
            testset = torchvision.datasets.SVHN(root='data', download=True, transform=transform_test)

        elif args.dataset == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)

        else:
            raise Exception("Invalid dataset name")

        if args.overfit:
            trainset = data.dataset.Subset(trainset, range(args.overfit_num_pts))
            testset = data.dataset.Subset(testset, range(args.overfit_num_pts))

        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # Real-NVP Model
        print('building realnvp model..')
        net = RealNVP(num_scales=args.num_scales, in_channels=3, mid_channels=64, num_blocks=args.num_blocks)

    elif args.model == 'pairednvp':

        # TODO: Datasets used are hardcoded for now. Maybe reverse MNIST and SVHN?
        trainset_x = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
        testset_x = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_test)

        trainset_x2 = torchvision.datasets.SVHN(root='data', download=True, transform=transform_train)
        testset_x2 = torchvision.datasets.SVHN(root='data', download=True, transform=transform_test)

        if args.overfit:
            trainset_x = data.dataset.Subset(trainset_x, range(args.overfit_num_pts))
            testset_x = data.dataset.Subset(testset_x, range(args.overfit_num_pts))

            trainset_x2 = data.dataset.Subset(trainset_x2, range(args.overfit_num_pts))
            testset_x2 = data.dataset.Subset(testset_x2, range(args.overfit_num_pts))

        trainloader_x = data.DataLoader(trainset_x, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers)
        testloader_x = data.DataLoader(testset_x, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers)

        trainloader_x2 = data.DataLoader(trainset_x2, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers)
        testloader_x2 = data.DataLoader(testset_x2, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers)

        # Paired-NVP Model
        print('Building pairednvp model..')
        net = PairedNVP(num_scales=args.num_scales, in_channels=3, mid_channels=64, num_blocks=args.num_blocks)
    else:
        raise Exception("Invalid model name")

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']

    loss_fns = [RealNVPLoss(lambda_max=lm) for lm in [args.lambda_max, 2 * args.lambda_max]]
    param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr)

    # TODO: in paired NVP setting, make X and X2 examples alternate batch-by-batch instead of epoch-by-epoch
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        if args.model == 'realnvp':
            train(epoch, net, trainloader, device, optimizer, loss_fns, args.max_grad_norm, len(trainloader.dataset))
            test(epoch, net, testloader, device, loss_fns, args.num_samples, args.num_epoch_samples,
                 len(testloader.dataset))
        elif args.model == 'pairednvp':
            paired_train_loader = alternate(trainloader_x, trainloader_x2)
            paired_test_loader = alternate(testloader_x, testloader_x2)

            # Hardcoded to X being before X2. Spits out False, True, False, True... forever
            is_double_flow_iter = alternate(iter(lambda: False, 2), iter(lambda: True, 2))

            num_train_examples = min(len(trainloader_x.dataset), len(trainloader_x2.dataset))*2
            train(epoch, net, paired_train_loader, device, optimizer, loss_fns, args.max_grad_norm,
                  num_train_examples, args.model, is_double_flow_iter=is_double_flow_iter)

            num_test_examples = min(len(testloader_x.dataset), len(testloader_x2.dataset))*2
            test(epoch, net, paired_test_loader, device, loss_fns, args.num_samples, args.num_epoch_samples,
                 num_test_examples, args.model, is_double_flow_iter=is_double_flow_iter)

        else: raise Exception('Invalid model name')


def train(epoch, net, trainloader, device, optimizer, loss_fns, max_grad_norm,
          num_examples, model='realnvp', is_double_flow_iter=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=num_examples) as progress_bar:
        if is_double_flow_iter is not None:
            loader = zip(trainloader, is_double_flow_iter)
        else:
            loader = trainloader
        for batch in loader:
            if is_double_flow_iter is not None:
                (x, _), double_flow = batch
            else:
                x, _ = batch
                double_flow = None
            x = x.to(device)
            optimizer.zero_grad()
            if model == 'realnvp':
                z, sldj = net(x, reverse=False)
            elif model == 'pairednvp':
                z, sldj = net(x, double_flow, reverse=False)

            single_loss_fn, double_loss_fn = loss_fns
            loss_fn = double_loss_fn if double_flow else single_loss_fn
            model_loss, jacobian_loss = loss_fn(z, sldj)

            print("Train losses: {} model loss; {} Jacobian clamp loss".format(model_loss, jacobian_loss))
            loss = model_loss + jacobian_loss
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))


def sample(net, batch_size, device, model='realnvp'):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
        model (str): Type of model
        double_flow (bool): For Paired-NVP, whether or not to sample from x or x2
    """
    z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)

    if model == 'realnvp':
        x, _ = net(z, reverse=True)
    elif model == 'pairednvp':
        x, _ = net(z, False, reverse=True)
        x2, _ = net(z, True, reverse=True)
        x = torch.cat((x,x2),dim=0)

    x = torch.sigmoid(x)
    return x


def test(epoch, net, testloader, device, loss_fns, num_samples, num_epoch_samples,
         num_examples, model='realnvp', is_double_flow_iter=None):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=num_examples) as progress_bar:
            if is_double_flow_iter is not None:
                loader = zip(testloader, is_double_flow_iter)
            else:
                loader = testloader
            for batch in loader:
                if is_double_flow_iter is not None:
                    (x, _), double_flow = batch
                else:
                    x, _ = batch
                    double_flow = None
                x = x.to(device)
                if model == 'realnvp':
                    z, sldj = net(x)
                elif model == 'pairednvp':
                    z, sldj = net(x, double_flow)
                single_loss_fn, double_loss_fn = loss_fns
                loss_fn = double_loss_fn if double_flow else single_loss_fn
                model_loss, jacobian_loss = loss_fn(z, sldj)
                print("Test losses: {} model loss; {} Jacobian clamp loss".format(model_loss, jacobian_loss))
                loss = model_loss + jacobian_loss
                loss_meter.update(loss.item(), x.size(0))
                progress_bar.set_postfix(loss=loss_meter.avg,
                                         bpd=util.bits_per_dim(x, loss_meter.avg))
                progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg

    if epoch % num_epoch_samples == 0:
        # Save samples and data
        images = sample(net, num_samples, device, model)
        os.makedirs('samples', exist_ok=True)
        images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
        torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

    # args for overfit/not, #scales/maybe other architecture stuff
    # arg for dataset

    parser.add_argument('--overfit', action='store_true', help='Constrain number of train/test points?')
    parser.add_argument('--overfit_num_pts', default=128, type=int, help='Number of points to use for overfitting')
    parser.add_argument('--dataset', default='MNIST', type=str, help='Which to use: e.g. MNIST, SVHN')
    parser.add_argument('--num_scales', default=3, type=int, help='Number of scales for model architecture')
    parser.add_argument('--num_blocks', default=8, type=int, help='Number of residual blocks')
    parser.add_argument('--num_epoch_samples', default=1, type=int, help='Sample per num_epoch_samples epochs')
    parser.add_argument('--model', default='realnvp', type=str, help='Type of model (realnvp or pairednvp)')
    parser.add_argument('--lambda_max', default=float('inf'), type=float, help='Jacobian clamping threshold')

    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
    parser.add_argument('--num_epochs', default=10000, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')

    best_loss = 1e20

    main(parser.parse_args())
