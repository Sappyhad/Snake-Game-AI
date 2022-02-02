import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # Prediction
    def forward(self, x):  # x is tensor
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    #zakładamy w kodzie że modele będą w tym danym folderze, w QT można będzie to może zmienić więc też do poprawki
    def save(self, file_name='model1.pth'):
        #czyli folder też będzie podawany z GUI
        mode_folder_path = './model'
        if not os.path.exists(mode_folder_path):
            os.makedirs(mode_folder_path)

        file_name = os.path.join(mode_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model1.pth'):
        mode_folder_path = './model'
        file_name = os.path.join(mode_folder_path, file_name)

        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            self.train()


class QTrainer:
    # lr = learning rate
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()  # criterion = loss function = (Q_new - Q)**2

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # its one dimensional, but we need it as: (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for index in range(len(game_over)):
            Q_new = reward[index]
            if not game_over[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))

            target[index][torch.argmax(action).item()] = Q_new

        # 2: r + y * max(next_prediction Q value)
        # pred.clone()
        # predictions[atgmax(actions)] = Q_new

        self.optimizer.zero_grad()
        loss = self.loss(target, pred)
        loss.backward()

        self.optimizer.step()
