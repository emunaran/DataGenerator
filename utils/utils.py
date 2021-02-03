import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def save_model(discriminator_critic, generator, args):
    model_path = os.path.join(os.getcwd(), 'save_model')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    checkpoint_path = os.path.join(model_path, f'{args.algorithm}_checkpoint_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth')

    save_checkpoint({
        'discriminator_critic': discriminator_critic.state_dict(),
        'generator': generator.state_dict(),
        'args': args,
    }, filename=checkpoint_path)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_model(args, discriminator_critic, generator):
    saved_checkpoint_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
    checkpoint = torch.load(saved_checkpoint_path)

    discriminator_critic.load_state_dict(checkpoint['discriminator_critic'])
    generator.load_state_dict(checkpoint['generator'])

    saved_args = checkpoint['args']
    saved_args.train = args.train

    return saved_args, discriminator_critic, generator


def visualize_results(real_data: pd.DataFrame, fake_data: pd.DataFrame, algorithm: str):

    images_path = os.path.join(os.getcwd(), 'images')
    if not os.path.isdir(images_path):
        os.makedirs(images_path)

    images_alg_path = os.path.join(os.getcwd(), f'images/{algorithm}')
    if not os.path.isdir(images_alg_path):
        os.makedirs(images_alg_path)

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
        plt.savefig(os.path.join(images_alg_path, name + '_distribution.png'))
        # plt.show()
        plt.clf()


def softmax2onehot(m: np.ndarray):
    a = np.argmax(m, axis=1)
    b = np.zeros(m.shape)
    b[np.arange(a.size), a]=1
    return b