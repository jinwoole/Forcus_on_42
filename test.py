'''
Tetris game using pygame
'''

import pygame
import random
import time
import sys

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 20
HEIGHT = 20

# This sets the margin between each cell

MARGIN = 5

# Create a 2 dimensional array. A two dimensional
# array is simply a list of lists.
grid = []
for row in range(20):
	# Add an empty array that will hold each cell
	# in this row
	grid.append([])
	for column in range(10):
		grid[row].append(0)  # Append a cell

# Initialize pygame
pygame.init()

# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [255, 505]
screen = pygame.display.set_mode(WINDOW_SIZE)

# Set title of screen
pygame.display.set_caption("Tetris")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# -------- Main Program Loop -----------
while not done:
	for event in pygame.event.get():  # User did something
		if event.type == pygame.QUIT:  # If user clicked close
			done = True  # Flag that we are done so we exit this loop

	# Set the screen background
	screen.fill(BLACK)

	# Draw the grid
	for row in range(20):
		for column in range(10):
			color = WHITE
			if grid[row][column] == 1:
				color = RED
			pygame.draw.rect(screen,
							 color,
							 [(MARGIN + WIDTH) * column + MARGIN,
							  (MARGIN + HEIGHT) * row + MARGIN,
							  WIDTH,
							  HEIGHT])

	# Limit to 60 frames per second
	clock.tick(60)
	# Go ahead and update the screen with what we've drawn.
	pygame.display.flip()

# Be IDLE friendly. If you forget this line, the program will 'hang'
# on exit.
pygame.quit()