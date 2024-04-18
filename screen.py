import pygame
from pygame.locals import *

class Screen:
    def __init__(self, filepath):
        pygame.init()
        # Initialize Pygame screen
        self.screen = pygame.display.set_mode((1880, 920), pygame.RESIZABLE)
        pygame.display.set_caption('Aqua Flora Forecast')
        self.clock = pygame.time.Clock()
        self.running = True
        self.folderpath = filepath
        # Set font sizes
        self.big_font_size = 20
        self.small_font_size = 15

    def update_input(self, category, light, temperature, humidity, soil_humidity, days):
        # Update input data
        self.category = category
        self.humidity = humidity
        self.temperature = temperature
        self.soil_humidity = soil_humidity
        self.light = light
        self.water_days = days

    def load_resources(self):
        # Load fonts and images
        self.big_font = pygame.font.Font('Sunny Spells Basic.ttf', self.big_font_size)
        self.small_font = pygame.font.Font('Sunny Spells Basic.ttf', self.small_font_size)
        self.img = pygame.image.load(self.folderpath)
        # Scale image to fit screen
        self.scaled_image = pygame.transform.scale(self.img,
                                                   (self.screen.get_width() // 3.5, self.screen.get_height() // 2))
        self.rect_image = self.scaled_image.get_rect(
            center=(self.screen.get_width() // 4, self.screen.get_height() - self.screen.get_height()//3))
        # Create rectangles for text rendering
        self.rect = Rect(500, 0, self.screen.get_width() // 3.5 + 50, self.screen.get_height() // 2 + 50)
        self.rect.center = (self.screen.get_width() // 4, self.screen.get_height() - self.screen.get_height()//3)

    def draw_scene(self):
        """Draws the game scene on the screen."""
        # Draw background
        self.screen.fill("darkgreen")
        # Draw white rectangle
        pygame.draw.rect(self.screen, 'white', self.rect)
        # Draw image
        self.screen.blit(self.scaled_image, self.rect_image)
        # Draw text
        self.screen.blit(self.top_text, self.rect_top_text)
        self.screen.blit(self.humidity_text, self.rect_humidity_text)
        self.screen.blit(self.temperature_text, self.rect_temperature_text)
        self.screen.blit(self.soil_humidity_text, self.rect_soil_humidity_text)
        self.screen.blit(self.light_text, self.rect_light_text)

    def render_text(self):
        """Renders the text for the game scene."""
        # Adjust font sizes based on screen width
        self.small_font_size = int(self.screen.get_width() / 23)
        self.big_font_size = int(self.screen.get_width() / 20)

        # Render text
        self.top_text = self.big_font.render(f'Your {self.category} needs water in {self.water_days} days', True,
                                             'white')
        self.humidity_text = self.small_font.render(f'Humidity: {self.humidity}%', True, 'white')
        self.temperature_text = self.small_font.render(f'Temperature: {self.temperature}°C', True, 'white')
        self.soil_humidity_text = self.small_font.render(f'Soil humidity: {self.soil_humidity}%', True, 'white')
        self.light_text = self.small_font.render(f'Light: {self.light}%', True, 'white')

        # Set positions for text rendering
        self.rect_top_text = self.top_text.get_rect(
            center=(self.screen.get_width()/2, self.screen.get_height() // 5))
        rect_data_text_centerx = self.screen.get_width() - self.screen.get_width() // 3.5
        self.rect_humidity_text = self.humidity_text.get_rect(
            center=(rect_data_text_centerx, self.rect_top_text.centery + 3 * self.small_font_size))
        self.rect_temperature_text = self.temperature_text.get_rect(
            center=(rect_data_text_centerx, self.rect_humidity_text.centery + self.small_font_size))
        self.rect_soil_humidity_text = self.soil_humidity_text.get_rect(
            center=(rect_data_text_centerx, self.rect_temperature_text.centery + self.small_font_size))
        self.rect_light_text = self.light_text.get_rect(
            center=(rect_data_text_centerx, self.rect_soil_humidity_text.centery + self.small_font_size))

    def main_loop(self, category, humidity, temperature, soil_humidity, light, days):
        # Load resources
        self.load_resources()

        # Event handling
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

        # Update input data
        self.update_input(category, humidity, temperature, soil_humidity, light, days)

        # Render text
        self.render_text()

        # Draw scene
        self.draw_scene()

        # Update display
        pygame.display.flip()

        # Cap frame rate
        self.clock.tick(80)

        # Print plant data
        print(f"Day: {days}, Light intensity: {light} lux, Temperature: {temperature}°C, Humidity: {humidity}%, Soil Humidity: {soil_humidity}%")
