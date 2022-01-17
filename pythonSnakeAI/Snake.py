import pygame
import random

# Init
pygame.init()

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
orange = (255, 165, 0)

# Window
width, height = 600, 400
game_display = pygame.display.set_mode((width, height))
pygame.display.set_caption('Snake Game')

message_font = pygame.font.SysFont("ubuntu", 30)
score_font = pygame.font.SysFont("ubuntu", 25)

# Clock
clock = pygame.time.Clock()

# Snake
snake_size = 10
snake_speed = 10

def draw_snake(snake_size, snake_pixels):
    for pixel in snake_pixels:
        pygame.draw.rect(game_display, white, [pixel[0], pixel[1], snake_size, snake_size])


# Score
def show_score(score):
    text = score_font.render("Score: " + str(score), True, orange)
    game_display.blit(text, [0,0])


def run_game():
    # INITIAL VALUES

    game_over = False
    game_close = False

    x = width/2
    y = height/2

    x_speed = 0
    y_speed = 0

    snake_pixels = []
    snake_length = 1

    food_x = round(random.randrange(0, width-snake_size)/10.0) * 10.0
    food_y = round(random.randrange(0, height-snake_size)/10.0) * 10.0

    # GAME EVENTS
    while not game_over:

        while game_close:
            game_display.fill(black)
            game_over_message1 = message_font.render("Game Over!", True, red)
            game_over_message2 = message_font.render("Press 1 to Quit", True, white)
            game_over_message3 = message_font.render("Press 2 to Restart", True, white)
            game_display.blit(game_over_message1, [width/3 + 20, height/3 + 20])
            game_display.blit(game_over_message2, [(width/3) - 100, (height/3) - 100])
            game_display.blit(game_over_message3, [(width/3) + 100, (height/3) - 100])

            show_score(snake_length - 1)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_2:
                        run_game()
                if event.type == pygame.QUIT:
                    game_over = True
                    game_close = False

        for event in pygame.event.get():
            # Game Over
            if event.type == pygame.QUIT:
                game_over = True
            # Is The Key Pushed
            if event.type == pygame.KEYDOWN:
                # Left
                if event.key == pygame.K_LEFT:
                    x_speed = -snake_size
                    y_speed = 0
                # Right
                if event.key == pygame.K_RIGHT:
                    x_speed = snake_size
                    y_speed = 0
                # Up
                if event.key == pygame.K_UP:
                    x_speed = 0
                    y_speed = -snake_size
                # Down
                if event.key == pygame.K_DOWN:
                    x_speed = 0
                    y_speed = snake_size

        # Boarder Collision
        if x >= width or x < 0 or y >= height or y < 0:
            game_close = True

        # Movement
        x += x_speed
        y += y_speed

    # DISPLAY
        # backgroud
        game_display.fill(black)
        # Snake init target
        pygame.draw.rect(game_display, orange, [food_x, food_y, snake_size, snake_size])
        # Adding pixels to snake for movement
        snake_pixels.append([x, y])
        # Deleting last pixel of snake's tail to make it move, but not grow
        if len(snake_pixels) > snake_length:
            del snake_pixels[0]
        # Checking if snake runs into itself, if yes = game over
        for pixel in snake_pixels[:-1]:
            if pixel == [x, y]:
                game_close = True

        draw_snake(snake_size, snake_pixels)
        show_score(snake_length - 1)

        pygame.display.update()
        # Food Collision
        if x == food_x and y == food_y:
            food_x = round(random.randrange(0, width - snake_size) / 10.0) * 10.0
            food_y = round(random.randrange(0, height - snake_size) / 10.0) * 10.0
            snake_length += 1

        clock.tick(snake_speed)

    pygame.quit()
    quit()

print(run_game())
