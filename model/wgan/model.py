import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_inputs, args, features_units_for_softmax: list = None):
        super(Generator, self).__init__()
        self.num_inputs = num_inputs
        self.features_units_for_softmax = features_units_for_softmax

        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc4 = nn.Linear(args.hidden_size, num_inputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        if self.features_units_for_softmax:
            fake_data = self.handel_categorical_features(x)
        else:
            fake_data = torch.tanh(self.fc4(x))
        return fake_data

    def handel_categorical_features(self, x):
        final_layer_tensors = []
        x = self.fc4(x)
        units_continuous, units_categorical = torch.split(x, [self.num_inputs - sum(self.features_units_for_softmax),
                                                              sum(self.features_units_for_softmax)], dim=1)
        final_layer_tensors.append(torch.tanh(units_continuous))
        if len(self.features_units_for_softmax) > 1:
            units_per_features = torch.split(units_categorical, self.features_units_for_softmax, dim=1)
            for section in units_per_features:
                final_layer_tensors.append(torch.softmax(section, dim=1))
        else:
            final_layer_tensors.append(torch.softmax(units_categorical, dim=1))
            # final_layer_tensors.append(torch.tensor(softmax2onehot(torch.softmax(units_categorical, dim=1).detach().cpu().numpy())).float())

        fake_data = torch.cat(final_layer_tensors, dim=1)
        return fake_data


class Critic(nn.Module):
    def __init__(self, num_inputs, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc4 = nn.Linear(args.hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        out = self.fc4(x)
        return out