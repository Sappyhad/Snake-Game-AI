import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from PlotHelper import plot
import os



MAX_MEMORY = 200_000  # Rozmiar pamięci maksymalnej
BATCH_SIZE = 2000  # Liczba sampli propagowanych przez sieć neuronową
LR = 0.001  # Learning Rate


def load_nog(file_name):
    '''
    Funkcja wczytująca liczbę gier z pliku txt modelu.
    '''
    file_name = file_name.split('/')
    file_name = file_name[-1]
    file_split = file_name.split(".")
    file = file_split[0]
    file = file + '.txt'
    file_path = "./files/"
    file = file_path + file
    if os.path.isfile(file) and os.stat(file).st_size > 0:
        with open(file, 'r') as f:
            x = f.readline()
            return int(x)
    else:
        return 0



class AI:
    '''
    Klasa AI odpowiadająca za implementację modelu do gry.
    '''
    def __init__(self, nog: int = 0):
        '''
        Funkcja z wartościami inicjującymi.
        :param nog: liczba rozegranych gier
        '''
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate MUST BE SMALLER THAN 1
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.number_of_games = nog

    def save_info(self, file_name, score, record):
        '''
        Funkcja zapisująca liczbę gier, score i rekord do pliku txt modelu.
        '''
        file_name = file_name.split('/')
        file_name = file_name[-1]
        file_split = file_name.split(".")
        file = file_split[0]
        file = file + '.txt'
        file_path = "./files/"
        file = file_path + file
        with open(file, 'w') as f:
            f.write(f'{str(self.number_of_games)}\n')
            f.write(f'{str(score)}\n')
            f.write(f'{str(record)}\n')
            f.close()


    # Get State
    def get_state(self, game):
        '''
        Funkcja okreslająca stan gry. Wszystkie parametry, po obliczeniu, kompresowane są do listy numpy.
        :param game: lista parametrów z game.py
        :return: stan gry w danym momencie
        '''
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, game_over):
        '''
        Funkcja zapamiętująca jeden cykl (krok) w grze.
        :param state: stan gry
        :param action: akcja
        :param reward: nagroda punktowa
        :param next_state: nastepny stan gry
        :param game_over: koniec gry
        '''
        self.memory.append((state, action, reward, next_state, game_over))  # pop left is MAX_MEMORY is reached

    def train_long_memory(self):
        '''
        Funkcja odpowiedzialna za zapamiętywanie długoterminowe (wiele cykli gier)
        '''
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        '''
        Funkcja odpowiedzialna za krótkoterminowe (jeden cykl gry)
        :param state: stan gry
        :param action: akcja
        :param reward: nagroda punktowa
        :param next_state: nastepny stan gry
        :param game_over: koniec gry
        '''
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        '''
        Funkcja podejmująca decyzje jaką akcję ma wykonać model.
        Wszystko na poczatku zależne jest od losowości (epsilon).
        :param state: stan gry
        :return: finalny akcja
        '''
        # random moves: tradeoff between exploration and exploitation in deep learning
        self.epsilon = 80 - self.number_of_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # raw value
            # getting max from prediction and converting to one number
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(file_name='model1.pth'):
    '''
    Funkcja odpowiedzialna za główną pętlę gry, pozyskiwanie parametrów do nauki modelu i rysowanie wykresu na bieżąco.
    :param file_name: nazwa pliku modelu
    '''
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = AI(load_nog(file_name))
    game = SnakeGameAI()

    agent.model.load(file_name)

    while True:
        # get old state
        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old)

        # perform the move and get new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # train long memory and plot result
            game.reset()
            agent.number_of_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(file_name)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_of_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            #zapis gier
            agent.save_info(file_name, score, record)



if __name__ == "__main__":
    train()
