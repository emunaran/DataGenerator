import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def save_model(discriminator, generator, args):
    model_path = os.path.join(os.getcwd(), 'save_model')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    checkpoint_path = os.path.join(model_path, 'checkpoint_' + '.pth.tar')

    save_checkpoint({
        'discriminator': discriminator.state_dict(),
        'generator': generator.state_dict(),
        'args': args,
    }, filename=checkpoint_path)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_model(args, discriminator_critic, generator):
    saved_checkpoint_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
    checkpoint = torch.load(saved_checkpoint_path)

    dis = discriminator_critic.load_state_dict(checkpoint['discriminator'])
    gen = generator.load_state_dict(checkpoint['generator'])
    return dis, gen, checkpoint['args']


def visualize_results(real_data: pd.DataFrame, fake_data: pd.DataFrame):

    images_path = os.path.join(os.getcwd(), 'images')
    if not os.path.isdir(images_path):
        os.makedirs(images_path)

    for name in real_data.columns:
        plt.xlabel('Values')
        plt.ylabel('Probability')
        plt.title(name + " distribution")

        # Collecting corresponding class values where the Outcome is 0/1 (Color red being 1 and blue being 0)
        fraud_dist = fake_data[name].values
        common_dist = real_data[name].values
        plt.hist(common_dist, 50, density=True, alpha=0.6, label='real')
        plt.hist(fraud_dist, 50, density=True, alpha=0.6, facecolor='r', label='fake')
        plt.legend()
        plt.savefig(os.path.join(images_path, name + '_distribution.png'))
        # plt.show()
        plt.clf()