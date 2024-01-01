# Pygame Draw Digits

# Imports
import pygame
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.neighbors import KNeighborsClassifier


class DrawDigit:
    def __init__(self):
        # APP Title
        pygame.display.set_caption("Draw PCA")

        # Window
        self.window_size = (560, 560)
        self.screen_app = pygame.display.set_mode(self.window_size)

        # Draw Variables
        self.drawing = False
        self.draw_points = []

        # Model and Eigeinvectors
        self.knc_model = joblib.load("./pca_model_eucl")
        self.top_eigenvectors = (np.load("./top_eigenvectors.npz"))["arr_0"]
        print(self.top_eigenvectors)
        print(self.top_eigenvectors.shape)

        # Font
        self.font = pygame.font.Font(None, 36)

    def run_app(self):
        while True:
            self.check_event()
            self.draw()

    def check_event(self):
        event = pygame.event.poll()

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # Mouse left button pressed
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.drawing = True
            self.draw_points = [event.pos]
        # Mouse left button pressed and drawing
        elif event.type == pygame.MOUSEMOTION and self.drawing:
            self.draw_points.append(event.pos)
        # Mouse left button released
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.drawing = False
            self.predict_entry()

    def draw(self):
        self.screen_app.fill((0, 0, 0))

        if len(self.draw_points) > 1:
            for center in self.draw_points:
                pygame.draw.circle(self.screen_app, (255, 255, 255), center, 15)

        pygame.display.flip()

    def predict_entry(self):
        image_screen = pygame.surfarray.array2d(self.screen_app)
        image_matrix = np.swapaxes(image_screen, 0, 1)
        downsize_image = zoom(image_matrix, 1 / 20, order=3)
        downsize_image_flat = (downsize_image.flatten()).reshape(1, -1)
        image_proj = np.dot(downsize_image_flat, self.top_eigenvectors)
        predicted_digit = self.knc_model.predict(image_proj)
        print(predicted_digit)


if __name__ == "__main__":
    pygame.init()
    DrawDigit().run_app()
