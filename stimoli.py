import pygame
import tkinter
from tkinter import simpledialog
import os
import socket
import sys
import math
import time
import random

'''GLOBAL VARIABLES AND CONSTANTS'''

# Dialogue window for the tester number
application_window = tkinter.Tk()
testernumber = simpledialog.askstring("Input", "Input tester number", parent=application_window)
#Insert the trial number of an user
index = simpledialog.askstring("Input", "Input trial number", parent=application_window)

# Color definitions
BLACK = (0, 0, 0)
RED = (255, 0, 0)


#Window creation
winsize = (sizeWidth, sizeHeight) = (1280, 720)
win_pos_left = 0
win_pos_top = 0

#Set environment variables
os.environ['SDL_VIDEO_WINDOW_POS'] = '{0},{1}'.format(win_pos_left, win_pos_top)
#initialize display for a window
screen = pygame.display.set_mode(winsize, pygame.FULLSCREEN)
pygame.display.set_caption("Test")
    
# Used to manage how fast the screen updates. Create an object to manage the time
clock = pygame.time.Clock()


'''SERVER CONNECTION'''

# Host machine IP
HOST = '127.0.0.1'
# Gazepoint Port
PORT = 4242
ADDRESS = (HOST, PORT)

# Connection
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(ADDRESS)

# Communication and server requests for the eye-tracker
s.send(str.encode('<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_POG_LEFT" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_POG_RIGHT" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_POG_BEST" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_PUPIL_LEFT" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_PUPIL_RIGHT" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_EYE_LEFT" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_EYE_RIGHT" STATE="1" />\r\n'))
s.send(str.encode('<SET ID="ENABLE_SEND_BLINK" STATE="1" />\r\n'))


def verticalMove(speed):
 
  #Starting image position
  x = (sizeWidth / 2) - (300/2)
  y = 0

  #Load and rescale the text image
  picture = pygame.image.load('Images/text2.png').convert()
  picture = pygame.transform.scale(picture, (300, 150))

  # File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\david\\OneDrive\\Desktop\\risultati\\Results{}-Trial{}.txt".format(testernumber, index), "w")
 
  # Setting the time
  test_time = 6
  t_end = time.time() + test_time
  wall = False

  while time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()
  
    # Sending data to the server and writing it on the respective file
    casual_data = s.recv(1024)
    file1.write(bytes.decode(casual_data))
    
    while not wall:
      if (y >= 0 and y < sizeHeight-150):
        y = y + (1 * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = True

    while wall:
      if(y > 1*speed):
        y = y - (1 * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = False
    
    pygame.display.flip()
    clock.tick(150)

  # Sending data to the server and writing it on the respective file
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
  time.sleep(0.3)
  casual_data = s.recv(1024)
  time.sleep(0.3)
  file1.write(bytes.decode(casual_data))
  file1.close()


def horizontalMove(speed):
 
  #Starting image position
  x = 0
  y = (sizeHeight / 2) - (150/2)

  #Load and rescale the text image
  picture = pygame.image.load('Images/text1.png').convert()
  picture = pygame.transform.scale(picture, (300, 150))

  # File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\david\\OneDrive\\Desktop\\risultati\\Results{}-Trial{}.txt".format(testernumber, index), "w")

  # Setting the time
  test_time = 6
  t_end = time.time() + test_time

  wall = False

  while time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()
    
    # Sending data to the server and writing it on the respective file
    casual_data = s.recv(1024)
    file1.write(bytes.decode(casual_data))

    while not wall:
      if (x >= 0 and x < sizeWidth-300):
        x = x + (1 * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = True

    while wall:
      if(x > 1*speed):
        x = x - (1 * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = False
    
    pygame.display.flip()
    clock.tick(150)

  # Sending data to the server and writing it on the respective file
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
  time.sleep(0.3)
  casual_data = s.recv(1024)
  time.sleep(0.3)
  file1.write(bytes.decode(casual_data))
  file1.close()
 



def concMove(speed):
 
  diag = math.sqrt((sizeWidth*sizeWidth)+(sizeHeight*sizeHeight))

  #Starting image position
  x = 0
  y = 0

  #Load and rescale the text image
  picture = pygame.image.load('Images/text3.png').convert()
  picture = pygame.transform.scale(picture, (300, 150))

  # File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\david\\OneDrive\\Desktop\\risultati\\Results{}-Trial{}.txt".format(testernumber, index), "w")

  # Setting the time
  test_time = 4
  t_end = time.time() + test_time

  wall = False

  while time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()
    
    # Sending data to the server and writing it on the respective file
    casual_data = s.recv(1024)
    file1.write(bytes.decode(casual_data))

    while not wall:
      if (x >= 0 and x < sizeWidth-300):
        x = x + (1 * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = True 

    while wall:
      if(y < sizeHeight-150):
        y = y + (1 * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = False 

    while not wall:
      if (x > 1):
        x = x - (1*(sizeWidth/diag) * speed)
        y = y - (1 *(sizeHeight/diag) * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = True

    while wall:
      if(y < sizeHeight-150):
        y = y + (1 * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = False

    while not wall:
      if (x < sizeWidth-300):
        x = x + (1*(sizeWidth/diag) * speed)
        y = y - (1 *(sizeHeight/diag) * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = True
    
    pygame.display.flip()
    clock.tick(150)

  # Sending data to the server and writing it on the respective file
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
  time.sleep(0.3)
  casual_data = s.recv(1024)
  time.sleep(0.3)
  file1.write(bytes.decode(casual_data))
  file1.close()


def main():
    
  pygame.init()
  pygame.mouse.set_visible(False)

  speed = 0.3

  #shuffle the order of the animations
  tests_list = [horizontalMove,concMove,verticalMove]
  random.shuffle(tests_list)

  #run the animation after the shuffle
  for funct in tests_list:
    funct(speed)

  pygame.quit()
  s.close()

if __name__ == "__main__":
    main()



