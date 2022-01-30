import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# RGB Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

SNAKE_SIZE = 20
SPEED = 40


class SnakeGameAI:
    # Constructor
    def __init__(self, w=640, h=480):
        # w-width
        # h-height
        self.w = w
        self.h = h
        # Initial display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    # Function for resetting to initial state of the game
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-SNAKE_SIZE, self.head.y),
                      Point(self.head.x-(2*SNAKE_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    # Steps of game actions
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collecting input from user
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Moving
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. Checking if game is over
        reward = 0
        game_over = False
        # 3.1 If snake collides -> reward = -10
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Placing new food or just moving
            # 4.1 If snake ate -> reward = 10 ?
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. Updating UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Returning game_over and score
        return reward, game_over, self.score

    # Is snake colliding with something
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.w - SNAKE_SIZE or pt.x < 0 or pt.y > self.h - SNAKE_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True

        return False

    # Placing the food on the game board
    def _place_food(self):
        # Getting random point
        x = random.randint(0, (self.w - SNAKE_SIZE) // SNAKE_SIZE) * SNAKE_SIZE
        y = random.randint(0, (self.h - SNAKE_SIZE) // SNAKE_SIZE) * SNAKE_SIZE
        # Setting food on place
        self.food = Point(x, y)
        # Checking if food is being placed on the snake. If so, recursively getting new random point
        if self.food in self.snake:
            self._place_food()

    # Updating UI -> help function
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, SNAKE_SIZE, SNAKE_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, SNAKE_SIZE, SNAKE_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    # Movement -> help function
    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Right turn (clock wise) r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Left turn (clock wise) r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += SNAKE_SIZE
        elif self.direction == Direction.LEFT:
            x -= SNAKE_SIZE
        elif self.direction == Direction.DOWN:
            y += SNAKE_SIZE
        elif self.direction == Direction.UP:
            y -= SNAKE_SIZE

        self.head = Point(x, y)
