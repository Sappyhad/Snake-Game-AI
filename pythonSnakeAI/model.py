import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    '''
    Klasa modelu sieci neuronowej.
    '''
    def __init__(self, input_size, hidden_size, output_size):
        '''
        Funkcja z inicjującymi wartościami modelu.
        '''
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        '''
        Funkcja obliczająca predykcję.
        :return: tensor
        '''
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


    def save(self, file_name='model1.pth'):
        '''
        Funkcja zapisująca plik modelu.
        '''
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model1.pth'):
        '''
        Funkcja wczytująca plik modelu.
        '''
        if os.path.exists(file_name) and os.stat(file_name).st_size > 0:
            self.load_state_dict(torch.load(file_name))
            self.train()


class QTrainer:
    '''
    Klasa nauki modelu.
    '''
    # lr = learning rate
    def __init__(self, model, lr, gamma):
        '''
        Funkcja z wartościami inicjującymi.
        :param model: typ modelu
        :param lr: wkaźnik nauczania
        :param gamma: parametr gamma
        '''
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()  # criterion = loss function = (Q_new - Q)**2

    def train_step(self, state, action, reward, next_state, game_over):
        '''
        Funkcja kroku nauczania. Krok składa się z nastepujących parametrów podawanych w argumentach.
        :param state: stan gry
        :param action: akcja podejmowana przez model
        :param reward: wynagrodzenie punktowe
        :param next_state: następny stan po wykonaniu określonych akcji
        :param game_over: koniec gry
        '''
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

        self.optimizer.zero_grad()
        loss = self.loss(target, pred)
        loss.backward()

        self.optimizer.step()
