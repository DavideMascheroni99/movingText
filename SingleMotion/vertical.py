import pygame
from tkinter import * 
from tkinter.ttk import *
import sys

def checkEnd():
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      sys.exit()
    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
      sys.exit()

def verticalMove(sizeWidth,sizeHeight, screen, black, speed):
 
 #Starting image position
 x = (sizeWidth / 2) - (300/2)
 y = 0

 #Load and rescale the text image
 picture = pygame.image.load('Images/text2.png').convert()
 picture = pygame.transform.scale(picture, (300, 150))

 finished = False
 wall = False
 count = 0

 while not finished:

  #Automatically stop after certain number of executions

  if(count > 2*speed):
    finished = True
  
  while not wall:
    if (y >= 0 and y < sizeHeight-150):
      y = y + (1 * speed)
      screen.fill(black)
      screen.blit(picture, (x,y))
      pygame.display.flip()
    else:
      wall = True

  checkEnd()

  while wall:
    if(y != 0):
      y = y - (1 * speed)
      screen.fill(black)
      screen.blit(picture, (x,y))
      pygame.display.flip()
    else:
      wall = False
    
  count = count + (1*speed)

  checkEnd()


def main():
    
    pygame.init()

    #Get screen size
    infoObject = pygame.display.Info()
    sizeWidth = infoObject.current_w
    sizeHeight = infoObject.current_h

    #Prepare and display the screen 
    screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)

    #background color
    black = pygame.color.Color('#000000') 
    #Speed definition
    speed = 0.5

    verticalMove(sizeWidth, sizeHeight, screen, black, speed)

    pygame.quit()

if __name__ == "__main__":
    main()


