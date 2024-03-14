import os
from utils.dataset import Loader, get_dataset
from utils.augment import EventAugment
import torch
from snn_utils.models import Classifier
import argparse
import wandb
import random
import numpy as np
from snn_utils.functions import TET_loss
import torch.distributed as dist
import torch.multiprocessing as mp


world_size = torch.cuda.device_count()
parser = argparse.ArgumentParser()
parser.add_argument('--l_mags', default=7, type=int, help='Number of magnitudes')
parser.add_argument('--train_num_workers', default=1, type=int)
parser.add_argument('--train_batch_size', default=4, type=int)
parser.add_argument('--train_num_epochs', default=100, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument("--train_distributed", type=bool, default=True)
parser.add_argument("--augment_mode", default="RPG", choices=["identity", "RPG", "eventdrop", "NDA"])
parser.add_argument('--weight_decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--classifier', default="vgg11", choices=["vgg11", "resnet34", "resnet18"])
parser.add_argument('--spiking_neuron', default="PLIF", choices=["LIF", "PLIF", "IF"])
parser.add_argument("--dataset", default="NCaltech101", choices=["CIFAR10DVS", "NCars", "NCaltech101", "DVSGesture", "miniNImageNet", "SLAnimals4sets", "SLAnimals3sets"])
parser.add_argument("--master_port", default="12121")
parser.add_argument("--mask", default="post")
parser.add_argument("--seed", default=0)
parser.add_argument("--use_wandb", default=False)
parser.add_argument("--timesteps", default=10, type=int)
parser.add_argument("--relcam_mask_std_ratio", default=1, type=float, help="the std ratio of pre-mask in RPGMix")
parser.add_argument("--relcam_coord_std_ratio", default=0, type=float, help="the std ratio of obtaining bounding boxes in RPGMix")
parser.add_argument("--relprop_mode", default="slrp", type=str, choices=["slrp", "sltrp"])
parser.add_argument("--relevance_mix", default="layer4", type=str, choices=["layer4", "long"])
parser.add_argument("--mix_prob", default=0.5, type=float, help="the max probability of mixing")
args = parser.parse_args()

if args.dataset == "NCaltech101":
    args.event_resolution = (180, 240)
    args.crop_dimension = (128, 128)
    args.num_classes = 101
    args.train_batch_size = 16
    args.lr = 1e-3
elif args.dataset == "CIFAR10DVS":
    args.event_resolution = (128, 128)
    args.crop_dimension = (48, 48)
    args.num_classes = 10
    args.train_batch_size = 64
    args.lr = 1e-3
elif args.dataset == "NCars":
    args.event_resolution = (100, 120)
    args.crop_dimension = (48, 48)
    args.num_classes = 2
    args.train_batch_size = 64
    args.lr = 1e-3
elif args.dataset == "DVSGesture":
    args.event_resolution = (128, 128)
    args.crop_dimension = (128, 128)
    args.num_classes = 11
    args.train_batch_size = 20
    args.timesteps = 16
    args.lr = 5e-4
    args.train_num_epochs = 200
elif args.dataset == "miniNImageNet":
    args.event_resolution = (480, 640)
    args.crop_dimension = (224, 224)
    args.num_classes = 100
    args.train_batch_size = 64
    args.lr = 1e-3
elif "SLAnimals" in args.dataset:
    args.event_resolution = (128, 128)
    args.crop_dimension = (128, 128)
    args.num_classes = 19
    args.train_batch_size = 20
    args.timesteps = 16
    args.lr = 5e-4
    args.train_num_epochs = 200
else:
    raise Exception("Dataset not found")

args.batch_size_per_gpu = args.train_batch_size // world_size

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True


if "resnet" in args.classifier:
    args.data_shape_prefix = "TB"
else:
    args.data_shape_prefix = "BT"

seed_everything(args.seed)

save_name = args.classifier + "_" + args.dataset + "_" + args.augment_mode
save_path = 'model/{}/SNN/{}'.format(args.dataset, save_name)
process_name = "SNN_" + save_name
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path = os.path.join(save_path, "model_state_dict.pth")

def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt)
    rt /= world_size
    return rt.item()

def mixup_criterion_raw_events(pred, target, tet_loss):
    if args.data_shape_prefix == "TB":  # T, B
        pred_mean = pred.mean(0)
    else:  # B, T
        pred_mean = pred.mean(1)
    sigma = target[:, 2]
    target = target[:, :2].to(torch.int64)
    loss = (sigma * tet_loss(pred, target[:, 0], mode=args.data_shape_prefix) + (1 - sigma) * tet_loss(pred, target[:, 1], mode=args.data_shape_prefix)).mean()
    accuracy = (pred_mean.argmax(1) == target[:, 0]).logical_or(pred_mean.argmax(1) == target[:, 1]).float().mean()
    return loss, accuracy


def distributed_train(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.master_port
    dist_url = "env://"
    device = torch.device(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method=dist_url,
                            world_size=world_size, rank=rank)
    train_ds, val_ds, test_ds = get_dataset(args)
    if rank == 0 and args.use_wandb:
        wandb.init(project="EventRPG", name=process_name, config=args)
    training_loader = Loader(train_ds, args, device, distributed=args.train_distributed, batch_size=args.batch_size_per_gpu, drop_last=True)
    validation_loader = Loader(val_ds, args, device, distributed=args.train_distributed, batch_size=args.batch_size_per_gpu)
    test_loader = Loader(test_ds, args, device, distributed=args.train_distributed, batch_size=args.batch_size_per_gpu)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    print("check 1")
    model = Classifier(voxel_dimension=(args.timesteps, *args.event_resolution), device=device, crop_dimension=args.crop_dimension, relprop_mode=args.relprop_mode,
                       classifier=args.classifier, num_classes=args.num_classes, pretrained=True, spiking_neuron=args.spiking_neuron).to(device)
    print("check 2")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    print("check 3")
    event_augment = EventAugment(args.event_resolution, model.module, l_mags=args.l_mags, mask_std_ratio=args.relcam_mask_std_ratio,
                                 coord_std_ratio=args.relcam_coord_std_ratio, relevance_mix=args.relevance_mix,
                                 data_shape_prefix=args.data_shape_prefix, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_num_epochs, eta_min=0.)
    best_val_accuracy = 0
    for epoch in range(args.train_num_epochs):
        sum_accuracy = 0
        sum_loss = 0
        training_loader.sampler.set_epoch(epoch)
        for events, labels in training_loader:
            spatial_mixup = False
            if args.augment_mode == 'RPG':
                aug_events, labels, spatial_mixup = event_augment.batch_augment(events, labels, args.mask, mix_prob=args.mix_prob)
            elif args.augment_mode == 'NDA':
                aug_events = event_augment.nda_batch_augment(events)
                if random.random() < 0.5:
                    spatial_mixup = True
                    aug_events, labels = event_augment.cut_mix(aug_events, labels)
            elif args.augment_mode == 'eventdrop':
                aug_events = event_augment.eventdrop_batch_augment(events)
            else:
                aug_events = events
            model.train()
            optimizer.zero_grad()
            y = model(aug_events)
            if args.data_shape_prefix == "TB":  # T, B, C, H, W
                y_mean = y.mean(0)
            else:  # B, T, C, H, W
                y_mean = y.mean(1)
            if spatial_mixup:
                loss, accuracy = mixup_criterion_raw_events(y, labels, TET_loss)
            else:
                loss = TET_loss(y, labels, mode=args.data_shape_prefix)
                accuracy = (y_mean.argmax(1) == labels).float().mean()
            sum_accuracy += accuracy
            sum_loss += loss
            loss.backward()
            optimizer.step()
        training_accuracy = sum_accuracy.item() / len(training_loader)
        training_loss = sum_loss.item() / len(training_loader)
        print("Epoch {}, Training Accuracy {}".format(str(epoch), str(training_accuracy)))
        lr_scheduler.step()
        sum_correct = 0
        sum_all_samples = 0
        sum_loss = 0
        model.eval()
        for events, labels in validation_loader:
            with torch.no_grad():
                y = model(events)
                if args.data_shape_prefix == "TB":  # T, B, C, H, W
                    y_mean = y.mean(0)
                else:  # B, T, C, H, W
                    y_mean = y.mean(1)
                loss = cross_entropy_loss(y_mean, labels)
                correct = (y_mean.argmax(1) == labels).float().sum()
                all_samples = torch.tensor([len(labels)]).float().to(rank)
            sum_correct += correct
            sum_all_samples += all_samples
            sum_loss += loss
        validation_loss = sum_loss / len(validation_loader)
        sum_correct = reduce_mean(sum_correct, 1)
        sum_all_samples = reduce_mean(sum_all_samples, 1)
        validation_accuracy = sum_correct / sum_all_samples
        validation_loss = reduce_mean(validation_loss, world_size)
        print("Epoch {}, Validating Accuracy {}, Validating Loss {}".format(str(epoch), str(validation_accuracy),
                                                                            str(validation_loss)))
        if best_val_accuracy < validation_accuracy and rank == 0:
            best_val_accuracy = validation_accuracy
            torch.save(model.module.classifier.state_dict(), save_path)
        sum_accuracy = 0
        sum_loss = 0
        for events, labels in test_loader:
            with torch.no_grad():
                y = model(events)
                y_mean = y.mean(1)
                loss = cross_entropy_loss(y_mean, labels)
                accuracy = (y_mean.argmax(1) == labels).float().mean()
            sum_accuracy += accuracy
            sum_loss += loss
        test_loss = sum_loss / len(test_loader)
        test_accuracy = sum_accuracy / len(test_loader)
        test_loss = reduce_mean(test_loss, world_size)
        test_accuracy = reduce_mean(test_accuracy, world_size)
        print("Epoch {}, test Accuracy {}, test loss {}".format(str(epoch), str(test_accuracy), str(test_loss)))
        if rank == 0 and args.use_wandb:
            wandb.log({"training/accuracy": training_accuracy,
                    "training/loss": training_loss,
                    "validating/accuracy": validation_accuracy,
                    "validating/loss": validation_loss,
                    "test/accuracy": test_accuracy,
                    "test/loss": test_loss})

    torch.cuda.empty_cache()

if __name__ == "__main__":
    mp.spawn(distributed_train, nprocs=world_size, join=True)