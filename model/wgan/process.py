from model.wgan.model import Generator, Critic
from tensorboardX import SummaryWriter
import numpy as np
from torch import optim
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
from utils.utils import *


class WGANProcess():
    def __init__(self, args, scaled_real_data: pd.DataFrame, features_units_for_softmax: list = None):
        self.args = args
        self.scaled_real_data = scaled_real_data
        self.features_dim = self.scaled_real_data.shape[1]
        self.generator = Generator(self.features_dim, self.args, features_units_for_softmax)
        self.critic = Critic(self.features_dim, self.args)

        # Additional WGAN params
        self.weight_clip = 0.01

    def run(self):

        if torch.cuda.is_available():
            self.critic.cuda()
            self.generator.cuda()
        critic_optim = optim.RMSprop(self.critic.parameters(), lr=self.args.learning_rate)
        generator_optim = optim.RMSprop(self.generator.parameters(), lr=self.args.learning_rate)
        data_loader = DataLoader(self.scaled_real_data.values, batch_size=self.args.batch_size, shuffle=True)

        if self.args.load_model is not None:
            load_model(args=self.args, discriminator_critic=self.critic, generator=self.generator)

        if self.args.train:
            writer = SummaryWriter(self.args.logdir)

            critic_loss_list = []
            generator_loss_list = []

            for epoch in range(self.args.epochs):
                if (epoch % 100 == 0):
                    print("Epoch ", epoch)

                for n_batch, real_batch in enumerate(data_loader):
                    real_data = Variable(real_batch).float()
                    if torch.cuda.is_available():
                        real_data = real_data.cuda()

                    critic_loss = self.train_critic(critic_optim, real_data)
                    generator_loss = self.train_generator(generator_optim, real_data)

                    critic_loss_list.append(critic_loss)
                    generator_loss_list.append(generator_loss)

                print(f'critic_loss: {np.mean(critic_loss_list)}, generator_loss: {np.mean(generator_loss_list)}')
                writer.add_scalar('log/critic_loss', float(np.mean(critic_loss_list)), epoch)
                writer.add_scalar('log/generator_loss', float(np.mean(generator_loss_list)), epoch)

            save_model(self.critic, self.generator, self.args)

        fake_data = self.generator.forward(torch.Tensor(np.random.uniform(-1, 1, self.scaled_real_data.shape))).detach().cpu().numpy()
        fake_data = pd.DataFrame(fake_data, columns=self.scaled_real_data.columns)
        # visualize_results(real_data=self.scaled_real_data, fake_data=fake_data)
        return fake_data

    def train_critic(self, critic_optim, real_data):
        fake_data = self.generator.forward(torch.Tensor(np.random.uniform(-1, 1, self.scaled_real_data.shape)))
        fake_data = torch.Tensor(fake_data)

        for _ in range(self.args.critic_update_num):
            critic_output_fake = self.critic(fake_data)
            real_data = torch.Tensor(real_data)
            critic_output_real = self.critic(real_data)

            critic_loss = -(torch.mean(critic_output_real) - torch.mean(critic_output_fake))

            critic_optim.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_optim.step()

            for p in self.critic.parameters():
                p.data.clamp_(-self.weight_clip, self.weight_clip)

        return critic_loss.detach().cpu().numpy()

    def train_generator(self, generator_optim, real_data):
        noise = torch.Tensor(np.random.uniform(-1, 1, real_data.shape))

        for _ in range(self.args.generator_update_num):

            fake_data = self.generator.forward(noise)
            fake_data = torch.Tensor(fake_data)
            critic_output_fake = self.critic(fake_data)

            generator_loss = -torch.mean(critic_output_fake)

            generator_optim.zero_grad()
            generator_loss.backward(retain_graph=True)
            generator_optim.step()

        return generator_loss.detach().cpu().numpy()