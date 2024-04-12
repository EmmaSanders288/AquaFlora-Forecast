import pygame
from pygame.locals import *

class Screen:
    def __init__(self,filepath):
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption('Aqua Flora Forecast')
        self.clock = pygame.time.Clock()
        self.running = True
        self.folderpath = filepath
        self.small_font_size = 60
        self.big_font_size = 56




    def update_input(self, category, humidity, temperature, soil_humidity, light, days):

        self.category = category
        self.humidity = humidity
        self.temperature = temperature
        self.soil_humidity = soil_humidity
        self.light = light
        self.water_days = days



    def load_resources(self):
        self.img = pygame.image.load(self.folderpath)
        self.scaled_image = pygame.transform.scale(self.img, (self.screen.get_width() // 3, self.screen.get_height() // 1.5))
        self.rect_image = self.scaled_image.get_rect(center=(self.screen.get_width() // 4, self.screen.get_height() // 2))

        self.rect = Rect(500, 0, self.screen.get_width() // 3 + 50, self.screen.get_height() // 1.5 + 50)
        self.rect.center = (self.screen.get_width() // 4, self.screen.get_height() // 2)

        self.big_font = pygame.font.Font('Sunny Spells Basic.ttf', self.big_font_size)
        self.small_font = pygame.font.Font('Sunny Spells Basic.ttf', self.small_font_size)

    def draw_scene(self):
        """Draws the game scene on the screen."""
        self.screen.fill("darkgreen")
        pygame.draw.rect(self.screen, 'white', self.rect)
        self.screen.blit(self.scaled_image, self.rect_image)
        self.screen.blit(self.top_text, self.rect_top_text)
        self.screen.blit(self.humidity_text, self.rect_humidity_text)
        self.screen.blit(self.temperature_text, self.rect_temperature_text)
        self.screen.blit(self.soil_humidity_text, self.rect_soil_humidity_text)
        self.screen.blit(self.light_text, self.rect_light_text)

    def render_text(self, category, humidity, temperature, soil_humidity, light, days):

        """Renders the text for the game scene."""
        self.top_text = self.big_font.render(f'Your {category} needs water in {days} days', True,
                                             'white')
        self.humidity_text = self.small_font.render(f'Humidity: {humidity}', True, 'white')
        self.temperature_text = self.small_font.render(f'Temperature: {temperature}', True, 'white')
        self.soil_humidity_text = self.small_font.render(f'Soil humidity: {soil_humidity}', True, 'white')
        self.light_text = self.small_font.render(f'Light: {light}', True, 'white')

        self.rect_top_text = self.top_text.get_rect(center=(self.screen.get_width() - self.screen.get_width() // 3.5, self.screen.get_height() // 4))
        self.rect_humidity_text = self.humidity_text.get_rect(center=(self.rect_top_text.centerx, self.rect_top_text.centery + 3 * self.small_font_size))
        self.rect_temperature_text = self.temperature_text.get_rect(center=(self.rect_top_text.centerx, self.rect_humidity_text.centery + self.small_font_size))
        self.rect_soil_humidity_text = self.soil_humidity_text.get_rect(center=(self.rect_top_text.centerx, self.rect_temperature_text.centery + self.small_font_size))
        self.rect_light_text = self.light_text.get_rect(center=(self.rect_top_text.centerx, self.rect_soil_humidity_text.centery + self.small_font_size))


    def main_loop(self, category, humidity, temperature, soil_humidity, light, days):
        self.load_resources()
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False


            self.render_text( category, humidity, temperature, soil_humidity, light, days)
            self.draw_scene()
            pygame.display.flip()
            self.clock.tick(60)


