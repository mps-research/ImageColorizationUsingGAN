from ray import tune
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from models import Generator, Discriminator, weights_init
from places365 import create_dataset, Places365
from config import config, datasets, netGs, netDs


class Trainable(tune.Trainable):
    def setup(self, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        dataset_config = datasets[config['dataset']]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = Places365(
            dataset_config['dst_dir'], train=True, transform=transform, target_transform=target_transform)
        self.train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

        val_dataset = Places365(
            dataset_config['dst_dir'], train=False, transform=transform, target_transform=target_transform)
        self.val_dataloader = DataLoader(val_dataset, batch_size=25, shuffle=True)

        self.fixed_gray_images, self.fixed_rgb_images = next(iter(self.val_dataloader))
        self.fixed_gray_images = self.fixed_gray_images.to(self.device)
        self.fixed_rgb_images = self.fixed_rgb_images.to(self.device)

        self.netG = Generator(**netGs[config['netG']]).to(self.device)
        self.netG.apply(weights_init)

        self.netD = Discriminator(**netDs[config['netD']], p=config['p']).to(self.device)
        self.netD.apply(weights_init)

        self.optimizerG = Adam(self.netG.parameters(), lr=config['lrG'], betas=(0.5, 0.999))
        self.optimizerD = Adam(self.netD.parameters(), lr=config['lrD'], betas=(0.5, 0.999))

        self.gan_criterion = nn.BCELoss()
        self.dis_criterion = nn.L1Loss()
        self.labmda = config['lambda']

        log_name = self.trial_id + '--' + \
            '--'.join(f'{key}-{value}' for key, value in config.items())
        self.writer = SummaryWriter(f'/logs/{log_name}')

        self.n_updates = 1

    def step(self):
        self.netG.train()
        self.netD.train()

        for gray_images, real_images in self.train_dataloader:
            self.netD.zero_grad()

            gray_images = gray_images.to(self.device)
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            labels = torch.full((batch_size, ), 1., dtype=torch.float, device=self.device)
            outputs = self.netD(real_images).view(-1)
            errD_real = self.gan_criterion(outputs, labels)
            errD_real.backward()

            fake_images = self.netG(gray_images)
            labels.fill_(0.)
            outputs = self.netD(fake_images.detach()).view(-1)
            errD_fake = self.gan_criterion(outputs, labels)
            errD_fake.backward()

            self.optimizerD.step()

            errD = (errD_real + errD_fake) / 2.
            self.writer.add_scalar('Discriminator/Loss', errD.item(), self.n_updates)

            self.netG.zero_grad()

            labels.fill_(1.)
            outputs = self.netD(fake_images).view(-1)
            errG_gan = self.gan_criterion(outputs, labels)
            errG_dis = self.dis_criterion(fake_images, real_images)
            errG = errG_gan + self.labmda * errG_dis
            errG.backward()

            self.optimizerG.step()

            self.writer.add_scalar('Generator/Distance Loss', errG_dis.item(), self.n_updates)
            self.writer.add_scalar('Generator/GAN Loss', errG_gan.item(), self.n_updates)
            self.writer.add_scalar('Generator/Total Loss', errG.item(), self.n_updates)

            self.n_updates += 1

        self.netG.eval()
        self.netD.eval()

        fake_images = self.netG(self.fixed_gray_images)
        real_images = self.fixed_rgb_images

        fake_image_grid = make_grid(fake_images, normalize=True, value_range=(-1, 1), nrow=5)
        real_image_grid = make_grid(real_images, normalize=True, value_range=(-1, 1), nrow=5)

        self.writer.add_image(f'{self.trial_id}/Fake', fake_image_grid, self.iteration)
        self.writer.add_image(f'{self.trial_id}/Real', real_image_grid, self.iteration)

        return {
            'errG': errG.item(),
            'errG_GAN': errG_gan.item(),
            'errG_DIS': errG_dis.item(),
            'errD': errD.item()
        }


if __name__ == '__main__':
    for dataset_config in datasets.values():
        try:
            create_dataset(**dataset_config)
        except FileExistsError:
            pass

    tune.run(
        Trainable,
        stop={'training_iteration': 100},
        config=config,
        resources_per_trial={'gpu': 1, 'cpu': 1},
        num_samples=5
    )
