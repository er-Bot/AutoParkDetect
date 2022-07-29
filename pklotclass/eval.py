import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import argparse
import pklotclass.model as pkmodel
from .dataloader import ParkingLotDataset
from .trainer import Trainer

def run_eval(cfg):
    test_img_path = cfg['test_img_path']
    test_img_labels = cfg['test_img_labels']
    batch_size = cfg['batch_size']

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = ParkingLotDataset(test_img_path, test_img_labels, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model_cfg = cfg['model']
    model = getattr(pkmodel, model_cfg['type'])(**model_cfg['args'])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.1)

    trainer = Trainer(model, criterion, optimizer, test_dataloader, test_dataloader, test_dataloader, scheduler, cfg)

    trainer.test()

def main():
    parser = argparse.ArgumentParser(description='Training of parking lot classification model')
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    run_eval(cfg)

if __name__ == '__main__':
    main()