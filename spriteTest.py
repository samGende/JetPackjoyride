
import pygame

pygame.init()

screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption('sprites')

sprite_sheet_image = pygame.image.load('Barry_Steakfries.webp').convert_alpha()
smaller_image = pygame.transform.scale(sprite_sheet_image, (24, 24))


def get_image(sheet, width, height):
    image = pygame.Surface((width, height)).convert_alpha()
    image.blit(sheet, (0, 0), (0, 0, width, height))
    return image


frame_0 = get_image(sprite_sheet_image, 24, 24)


BG = (50, 50, 50)

run = True

while run:
    screen.fill(BG)

    screen.blit(smaller_image, (0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()
