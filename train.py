import os
from utils.augment import EventAugment
from utils.models import Classifier
from utils.dataset import Loader, get_dataset
import torch
import argparse
from tqdm import tqdm
import numpy as np
import random
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--l_mags', default=7, type=int, help='Number of magnitudes')
parser.add_argument('--train_num_workers', default=4, type=int)
parser.add_argument('--train_batch_size', default=4, type=int)
parser.add_argument('--train_num_epochs', default=100, type=int)
parser.add_argument('--lr', default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', '--wd', default=0, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--event_resolution', default=(128, 128), help='Resolution of events')
parser.add_argument("--augment_mode", default="RPG", choices=["identity", "RPG", "eventdrop", "NDA"])
parser.add_argument("--train_distributed", type=bool, default=False)
parser.add_argument('--validation_dataset', default=None)
parser.add_argument("--dataset", default="NCars", choices=["CIFAR10DVS", "NCars", "NCaltech101", "DVSGesture", "miniNImageNet", "SLAnimals4sets", "SLAnimals3sets"])
parser.add_argument("--representation", default="EST", choices=["VoxelGrid", "EST"])
parser.add_argument("--classifier", default="vgg19", choices=["vgg19", "resnet34", "resnet18"])
parser.add_argument("--timesteps", default=10)
parser.add_argument("--use_wandb", default=False)
parser.add_argument("--relcam_mask_std_ratio", default=1, type=float, help="the std ratio of pre-mask in RPGMix")
parser.add_argument("--relcam_coord_std_ratio", default=0, type=float, help="the std ratio of obtaining bounding boxes in RPGMix")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--master_port", default="12456")
parser.add_argument("--relevance_mix", default="long", type=str, choices=["layer4", "long"])
parser.add_argument("--mask", default="no")
parser.add_argument("--mix_prob", default=0.5, type=float, help="the max probability of mixing")
args = parser.parse_args()

if args.dataset == "NCaltech101":
    args.event_resolution = (180, 240)
    args.crop_dimension = (240, 240)
    args.num_classes = 101
    args.train_batch_size = 16
    args.lr = 1e-4
elif args.dataset == "CIFAR10DVS":
    args.event_resolution = (128, 128)
    args.crop_dimension = (224, 224)
    args.num_classes = 10
    args.train_batch_size = 64
    args.lr = 1e-4
elif args.dataset == "NCars":
    args.event_resolution = (100, 120)
    args.crop_dimension = (224, 224)
    args.num_classes = 2
    args.train_batch_size = 64
    args.lr = 1e-4
elif args.dataset == "DVSGesture":
    args.event_resolution = (128, 128)
    args.crop_dimension = (224, 224)
    args.num_classes = 11
    args.train_batch_size = 64
    args.timesteps = 16
    args.lr = 1e-4
    args.train_num_epochs = 200
elif args.dataset == "miniNImageNet":
    args.event_resolution = (480, 640)
    args.crop_dimension = (224, 224)
    args.num_classes = 100
    args.train_batch_size = 64
    args.lr = 1e-4
elif "SLAnimals" in args.dataset:
    args.event_resolution = (128, 128)
    args.crop_dimension = (224, 224)
    args.num_classes = 19
    args.train_batch_size = 128
    args.timesteps = 16
    args.lr = 1e-4
    args.train_num_epochs = 200
else:
    raise Exception("Dataset not found")

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


if 'RPG' in args.augment_mode:
    augment_mode = 'RPG'
elif 'eventdrop' in args.augment_mode:
    augment_mode = 'eventdrop'
elif 'NDA' in args.augment_mode:
    augment_mode = 'NDA'
else:
    augment_mode = 'origin'

seed_everything(args.seed)
world_size = torch.cuda.device_count()
save_name = args.classifier + "_" + args.dataset + "_" + args.augment_mode
process_name = args.representation + "_" + save_name
save_path = 'model/{}/{}/{}'.format(args.dataset, args.representation, save_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path = os.path.join(save_path, "model_state_dict.pth")
if args.use_wandb:
    wandb.init(project="EventRPG", name=process_name, config=args)


def mixup_criterion_raw_events(pred, target, cross_entropy_loss):
    sigma = target[:, 2]
    target = target[:, :2].to(torch.int64)
    loss = 0
    for i in range(pred.size(0)):
        loss += sigma[i] * cross_entropy_loss(pred[i].unsqueeze(0), target[i, 0].unsqueeze(0)) + (
                1 - sigma[i]) * cross_entropy_loss(pred[i].unsqueeze(0), target[i, 1].unsqueeze(0))
    loss = loss / pred.size(0)
    accuracy = (pred.argmax(1) == target[:, 0]).logical_or(pred.argmax(1) == target[:, 1]).float().mean()
    return loss, accuracy


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_dataset(args)
    model = Classifier(voxel_dimension=(args.timesteps, *args.event_resolution), num_classes=args.num_classes, event_representation=args.representation,
                       device=device, crop_dimension=args.crop_dimension, classifier=args.classifier, pretrained=True).to(device)
    event_augment = EventAugment(args.event_resolution, model, l_mags=args.l_mags,
                                 mask_std_ratio=args.relcam_mask_std_ratio, coord_std_ratio=args.relcam_coord_std_ratio,
                                 relevance_mix=args.relevance_mix, device=device)
    training_loader = Loader(train_ds, args, device, distributed=args.train_distributed, batch_size=args.train_batch_size)
    validation_loader = Loader(val_ds, args, device, distributed=args.train_distributed, batch_size=args.train_batch_size)
    test_loader = Loader(test_ds, args, device, distributed=args.train_distributed, batch_size=args.train_batch_size)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    best_test_accuracy = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_num_epochs, eta_min=0.)
    for i in range(args.train_num_epochs):
        sum_accuracy = 0
        sum_loss = 0
        for events, labels in tqdm(training_loader):
            B = int(1 + events[-1, -1].item())
            aug_events = []
            spatial_mixup = False
            if augment_mode == 'RPG':
                aug_events, labels, spatial_mixup = event_augment.batch_augment(events, labels, args.mask, args.mix_prob)
            elif augment_mode == 'NDA':
                aug_events = event_augment.nda_batch_augment(events)
                if random.random() < 0.5:
                    spatial_mixup = True
                    aug_events, labels = event_augment.cut_mix(aug_events, labels)
            elif augment_mode == 'eventdrop':
                aug_events = event_augment.eventdrop_batch_augment(events)
            else:
                aug_events = events
            model = model.train()
            optimizer.zero_grad()
            pred_labels = model(aug_events)
            if spatial_mixup:
                loss, accuracy = mixup_criterion_raw_events(pred_labels, labels, cross_entropy_loss)
            else:
                loss = cross_entropy_loss(pred_labels, labels)
                accuracy = (pred_labels.argmax(1) == labels).float().mean()
            sum_accuracy += accuracy
            sum_loss += loss
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        training_accuracy = sum_accuracy.item() / len(training_loader)
        training_loss = sum_loss.item() / len(training_loader)
        print("Epoch {}, Training Accuracy {}".format(str(i), str(training_accuracy)))

        sum_accuracy = 0
        sum_loss = 0
        model = model.eval()
        for events, labels in tqdm(validation_loader):
            with torch.no_grad():
                pred_labels = model(events)
                loss = cross_entropy_loss(pred_labels, labels)
                accuracy = (pred_labels.argmax(1) == labels).float().mean()
            sum_accuracy += accuracy
            sum_loss += loss
        validation_loss = sum_loss.item() / len(validation_loader)
        validation_accuracy = sum_accuracy.item() / len(validation_loader)
        print("Epoch {}, Validating Accuracy {}".format(str(i), str(validation_accuracy)))

        sum_accuracy = 0
        sum_loss = 0
        for events, labels in tqdm(test_loader):
            with torch.no_grad():
                pred_labels = model(events)
                loss = cross_entropy_loss(pred_labels, labels)
                accuracy = (pred_labels.argmax(1) == labels).float().mean()
            sum_accuracy += accuracy
            sum_loss += loss
        test_loss = sum_loss.item() / len(test_loader)
        test_accuracy = sum_accuracy.item() / len(test_loader)
        if best_test_accuracy < test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), save_path)
        print("Epoch {}, test Accuracy {}".format(str(i), str(test_accuracy)))
        if args.use_wandb:
            wandb.log({"training/accuracy": training_accuracy,
                       "training/loss": training_loss,
                       "validating/accuracy": validation_accuracy,
                       "validating/loss": validation_loss,
                       "test/accuracy": test_accuracy,
                       "test/loss": test_loss})
