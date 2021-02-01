from model.gan.model import Generator, Discriminator
from tensorboardX import SummaryWriter
import numpy as np
from torch import optim
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from utils.utils import *


class GANProcess():
    def __init__(self, args, scaled_real_data: pd.DataFrame):
        self.args = args
        self.scaled_real_data = scaled_real_data
        self.features_dim = self.scaled_real_data.shape[1]
        self.generator = Generator(self.features_dim, self.args)
        self.discriminator = Discriminator(self.features_dim, self.args)

    def run(self):

        if torch.cuda.is_available():
            self.discriminator.cuda()
            self.generator.cuda()
        discriminator_optim = optim.Adam(self.discriminator.parameters(), lr=self.args.learning_rate)
        generator_optim = optim.Adam(self.generator.parameters(), lr=self.args.learning_rate)
        data_loader = DataLoader(self.scaled_real_data.values, batch_size=self.args.batch_size, shuffle=True)

        if self.args.load_model is not None:
            load_model(args=self.args, discriminator_critic=self.discriminator, generator=self.generator)

        if self.args.train:
            writer = SummaryWriter(self.args.logdir)

            discrim_loss_list = []
            generator_loss_list = []

            for epoch in range(self.args.epochs):
                if (epoch % 100 == 0):
                    print("Epoch ", epoch)

                for n_batch, real_batch in enumerate(data_loader):
                    real_data = Variable(real_batch).float()
                    if torch.cuda.is_available():
                        real_data = real_data.cuda()

                    _, _, discrim_loss = self.train_discrim(discriminator_optim, real_data)
                    _, generator_loss = self.train_generator(generator_optim, real_data)

                    discrim_loss_list.append(discrim_loss)
                    generator_loss_list.append(generator_loss)

                print(f'critic_loss: {np.mean(discrim_loss_list)}, generator_loss: {np.mean(generator_loss_list)}')
                writer.add_scalar('log/critic_loss', float(np.mean(discrim_loss_list)), epoch)
                writer.add_scalar('log/generator_loss', float(np.mean(generator_loss_list)), epoch)

            save_model(self.discriminator, self.generator, self.args)

        fake_data = self.generator.forward(torch.Tensor(np.random.uniform(-1, 1, self.scaled_real_data.shape))).detach().cpu().numpy()
        fake_data = pd.DataFrame(fake_data, columns=self.scaled_real_data.columns)
        visualize_results(real_data=self.scaled_real_data, fake_data=fake_data)

    def train_discrim(self, discriminator_optim, real_data):
        fake_data = self.generator.forward(torch.Tensor(np.random.uniform(-1, 1, self.scaled_real_data.shape)))
        fake_data = torch.Tensor(fake_data)

        criterion = torch.nn.BCELoss()

        for _ in range(self.args.discrim_update_num):
            discrim_output_fake = self.discriminator(fake_data)
            real_data = torch.Tensor(real_data)
            discrim_output_real = self.discriminator(real_data)

            discrim_loss = criterion(discrim_output_fake, torch.ones((fake_data.shape[0], 1))) + \
                           criterion(discrim_output_real, torch.zeros((real_data.shape[0], 1)))

            discriminator_optim.zero_grad()
            discrim_loss.backward(retain_graph=True)
            discriminator_optim.step()

        discriminator_error = ((self.discriminator(real_data) < 0.5).float()).mean()
        generator_success = ((self.discriminator(fake_data) > 0.5).float()).mean()

        return discriminator_error, generator_success, discrim_loss.detach().cpu().numpy()

    def train_generator(self, generator_optim, real_data):
        noise = torch.Tensor(np.random.uniform(-1, 1, real_data.shape))
        # fake_data = self.generator.forward(torch.Tensor(np.random.uniform(-1, 1, real_data.shape)))
        # fake_data = torch.Tensor(fake_data)

        criterion = torch.nn.BCELoss()

        for _ in range(self.args.generator_update_num):

            fake_data = self.generator.forward(noise)
            fake_data = torch.Tensor(fake_data)
            discrim_output_fake = self.discriminator(fake_data)

            generator_loss = criterion(discrim_output_fake, torch.zeros((fake_data.shape[0], 1)))

            generator_optim.zero_grad()
            generator_loss.backward(retain_graph=True)
            generator_optim.step()

        generator_success = ((self.discriminator(fake_data) > 0.5).float()).mean()

        return generator_success, generator_loss.detach().cpu().numpy()