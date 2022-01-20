from ray import tune
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from models import Generator, Discriminator, weights_init
from datasets import Places365
from config import config, netGs, netDs


class Trainable(tune.Trainable):
    def setup(self, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

        target_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = Places365(
            '/data/places365_standard', train=True, transform=transform, target_transform=target_transform)
        self.train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

        val_dataset = Places365(
            '/data/places365_standard', train=False, transform=transform, target_transform=target_transform)
        self.val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=True)

        self.fixed_gray_images, self.fixed_rgb_images = next(iter(self.val_dataloader))
        self.fixed_gray_images = self.fixed_gray_images.to(self.device)
        self.fixed_rgb_images = self.fixed_rgb_images.to(self.device)

        self.netG = Generator(**netGs[config['netG']]).to(self.device)
        self.netG.apply(weights_init)

        self.netD = Discriminator(**netDs[config['netD']]).to(self.device)
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
            self.writer.add_scalar('Discriminator Loss', errD.item(), self.n_updates)

            self.netG.zero_grad()

            labels.fill_(1.)
            outputs = self.netD(fake_images).view(-1)
            errG_gan = self.gan_criterion(outputs, labels)
            errG_dis = self.dis_criterion(fake_images, real_images)
            errG = errG_gan + self.labmda * errG_dis
            errG.backward()

            self.optimizerG.step()

            self.writer.add_scalar('Generator Loss', errG.item(), self.n_updates)
            self.writer.add_scalar('Generator GAN Loss', errG_gan.item(), self.n_updates)
            self.writer.add_scalar('Generator Distance Loss', errG_dis.item(), self.n_updates)

            self.n_updates += 1

            if self.n_updates % 10000 == 0:
                self.netG.eval()
                self.netD.eval()

                fake_images = self.netG(self.fixed_gray_images)
                real_images = self.fixed_rgb_images

                images = torch.cat([fake_images, real_images])
                image_grid = make_grid(images, normalize=True, value_range=(-1, 1), nrow=10)
                self.writer.add_image('Images', image_grid, self.n_updates)

                self.netG.train()
                self.netD.train()

        return {
            'errG': errG.item(),
            'errG_GAN': errG_gan.item(),
            'errG_DIS': errG_dis.item(),
            'errD': errD.item()
        }


if __name__ == '__main__':
    tune.run(
        Trainable,
        stop={'training_iteration': 200},
        config=config,
        resources_per_trial={'gpu': 0.5, 'cpu': 1}
    )
