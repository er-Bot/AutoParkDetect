import torch
from torch.utils.data import Dataset
from torchvision import transforms
import yaml
import argparse
from pathlib import Path
import os
from PIL import Image
import pklotclass.model as pkmodel
from .log import create_log

family_dict = {0: 'Empty', 1: 'Occupied'}

class Predictor:
    def __init__(self, model, img_dataset, cfg):
        self.cfg = cfg
        self.save_dir = Path(self.cfg['location']) / 'eval'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = create_log('predict', self.save_dir, logformat='small', console=True)

        with open(self.save_dir / 'config.yml', 'w') as file:
            yaml.dump(cfg, file)

        self.img_dataset = img_dataset

        self.device, device_ids = self._prepare_device(self.cfg['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self._resume_checkpoint(self.cfg['location'] + '/model_best.pth')

    def predict(self):
        for idx, img in enumerate(self.img_dataset):
            img = img.to(self.device)
            predicted_label_vec = self.model(img)
            label = predicted_label_vec.argmax(1)
            family = family_dict[label]
            self.logger.info(f'Image #{idx+1:d} belongs to {family} family')

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info('Loading checkpoint: {} ...'.format(resume_path))

        checkpoint = torch.load(resume_path)
        self.model.load_state_dict(checkpoint['state_dict'])

        self.logger.info('Checkpoint loaded.')


class ParkingLotDataset(Dataset):
    def __init__(self, img_dir, img_path, transforms = None):
        with open(img_path, 'r') as f:
            lines = f.readlines()
            self.img_list = [os.path.join(img_dir, i.split()[0]) for i in lines]
            self.transforms = transforms

    def __getitem__(self, index):
        try:
            img_path = self.img_list[index]
            img = Image.open(img_path)
            img = self.transforms(img)
        except:
            return None
        return img

    def __len__(self):
        return len(self.label_list)


def run_predict(cfg):
    model_cfg = cfg['model']
    model = getattr(pkmodel, model_cfg['type'])(**model_cfg['args'])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_dir = cfg['img_dir']
    img_path = cfg['img_path']
    img_dataset = ParkingLotDataset(img_dir, img_path, transform)

    predictor = Predictor(model, img_dataset, cfg)

    predictor.predict()

def main():
    parser = argparse.ArgumentParser(description='Training of Protein classification model')
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    run_predict(cfg)

if __name__ == '__main__':
    main()