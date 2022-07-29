import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import yaml
import argparse

import pklotclass.model as pkmodel
from .dataloader import ParkingLotDataset
from .trainer import Trainer

def run_train(cfg):
    train_img_path = cfg['train_img_path']
    train_img_labels = cfg['train_img_labels']
    test_img_path = cfg['test_img_path']
    test_img_labels = cfg['test_img_labels']
    batch_size = cfg['batch_size']

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = ParkingLotDataset(train_img_path, train_img_labels, transform)
    if 'val_img_labels' in cfg:
        val_img_path = cfg['val_img_path']
        val_img_labels = cfg['val_img_labels']
        val_dataset = ParkingLotDataset(val_img_path, val_img_labels, transform)
    else:
        len_val = int(0.2 * len(train_dataset))
        len_train = len(train_dataset) - len_val
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (len_train, len_val))
    test_dataset = ParkingLotDataset(test_img_path, test_img_labels, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model_cfg = cfg['model']
    model = getattr(pkmodel, model_cfg['type'])(**model_cfg['args'])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, cfg['optimizer']['type'])(model.parameters(), **cfg['optimizer']['args'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **cfg['scheduler'])

    trainer = Trainer(model, criterion, optimizer, train_dataloader, val_dataloader, test_dataloader, scheduler, cfg)

    trainer.train()

    trainer.test()

def main():
    parser = argparse.ArgumentParser(description='Training of PKLot classification model')
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    run_train(cfg)

if __name__ == '__main__':
    main()