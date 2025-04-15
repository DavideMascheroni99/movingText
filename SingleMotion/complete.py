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

'''# Dialogue window for the tester number
application_window = tkinter.Tk()
testernumber = simpledialog.askstring("Input", "Input tester number", parent=application_window)
#Insert the trial number
index = simpledialog.askstring("Input", "Input trial number", parent=application_window)
'''
# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

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

'''# Host machine IP
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
s.send(str.encode('<SET ID="ENABLE_SEND_BLINK" STATE="1" />\r\n'))'''


#find the last point of a block
def find_last_point(text):
  point = text.rfind('.')
  return point


#add # every n characters
def addSeparator(text, n, max_lenght):
  s = list(text)
  ml = len(text)

  if(max_lenght > ml):
    max_lenght = ml

  for i in range(max_lenght):
    if(i%n == 0) and i != 0:
      if(s[i] == ' '):
        del(s[i])
      s[i] = '#' + s[i]
      
  text = ''.join(s[0:i])
  return text


#Finish the block text after the last point
def rem_text(text, n, max_lenght):
  txt = addSeparator(text, n, max_lenght)
  s = list(txt)
  point = find_last_point(txt)
  if (point != -1):
    point = point + 1
    for i in range(len(txt)):
      if (i == point):
        fin = ''.join(s[0:i])
        return fin
  return txt
  


def createVertBlock(x, y, font, nlText, char_size):

  for i in range(len(nlText)):
    img = font.render(nlText[i], True, WHITE)
    screen.blit(img, (x, y))
    y = y + char_size



#Box text vertical move
def horizontalMove():
  
  #dimension of a single character
  dim_char = 30
  #number of a box lines
  num_line = 6
  #Number of characters per line
  n = 35

  #Create a font
  font = pygame.font.SysFont("Arial", dim_char)
  #Text to show
  text = "Eclipses only occur when the Sun, Earth, and Moon are all in a straight line. Solar eclipses occur at new moon, when the Moon is between the Sun and Earth. In contrast, lunar eclipses occur at full moon, when Earth is between the Sun and Moon. The apparent size of the Moon is roughly the same as that of the Sun, with both being viewed at close to one-half a degree wide. The Sun is much larger than the Moon, but it is the vastly greater distance that gives it the same apparent size as the much closer and much smaller Moon from the perspective of Earth. The variations in apparent size, due to the non-circular orbits, are nearly the same as well, though occurring in different cycles. This makes possible both total (with the Moon appearing larger than the Sun) and annular (with the Moon appearing smaller than the Sun) solar eclipses.[217] In a total eclipse, the Moon completely covers the disc of the Sun and the solar corona becomes visible to the naked eye."
  # Setting the time
  test_time = 10
  t_end = time.time() + test_time

  #Starting image position and speed
  x = 0
  y = (sizeHeight / 2) - ((num_line*dim_char)/2)
  speed = 0.2

  text = rem_text(text, n, n*num_line)
  #text with new line
  nlText = text.split("#")
  max = 0
  #Get text width and height
  for ind in range(num_line-1):
    line_width, line_height = font.size(nlText[ind])
    if(line_width > max):
      max = line_width

  char_width = max/n


  '''
  # File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\Davide Mascheroni\\Desktop\\Risultati\\Results{}-Trial{}.txt".format(testernumber, index), "a")
  '''

  wall = False

  while time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()

    while not wall and time.time() <= t_end:

      '''# Sending data to the server and writing it on the respective file
      casual_data = s.recv(1024)
      file1.write(bytes.decode(casual_data))'''

      if (x >= 0 and x < sizeWidth - char_width*(n)):
        x = x + (1 * speed)
        screen.fill(BLACK)
        createVertBlock(x, y, font, nlText, dim_char)
        pygame.display.flip()
      else:
        wall = True

    while wall and time.time() <= t_end:

      '''# Sending data to the server and writing it on the respective file
      casual_data = s.recv(1024)
      file1.write(bytes.decode(casual_data))'''

      if(x > 1*speed):
        x = x - (1 * speed)
        screen.fill(BLACK)
        createVertBlock(x, y, font, nlText, dim_char)
        pygame.display.flip()
      else:
        wall = False
    
    pygame.display.flip()
    clock.tick(150)

  '''# Sending data to the server and writing it on the respective file
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
  time.sleep(0.3)
  casual_data = s.recv(1024)
  time.sleep(0.3)
  file1.write(bytes.decode(casual_data))
  file1.write("\n\n HorizontalMove end\n\n")
  file1.close()
  time.sleep(0.3)'''
 



#Box text horizontal move
def verticalMove():
 
  #Starting image position
  x = (sizeWidth / 2) - (300/2)
  y = 0
  speed = 0.2

  #Load and rescale the text image
  #picture = pygame.image.load('movingText/Images/text2.png').convert()
  picture = pygame.image.load('Programs/Images/text2.png').convert()
  picture = pygame.transform.scale(picture, (300, 150))

  '''# File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\Davide Mascheroni\\Desktop\\Risultati\\Results{}-Trial{}.txt".format(testernumber, index), "a")'''

  # Setting the time
  test_time = 10
  t_end = time.time() + test_time
  wall = False

  while time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()
  
    while not wall and time.time() <= t_end:

      '''# Sending data to the server and writing it on the respective file
      casual_data = s.recv(1024)
      file1.write(bytes.decode(casual_data))'''

      if(y >= 0 and y < sizeHeight-150):
        y = y + (1 * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = True

    while wall and time.time() <= t_end:

      '''# Sending data to the server and writing it on the respective file
      casual_data = s.recv(1024)
      file1.write(bytes.decode(casual_data))'''

      if(y > 1*speed):
        y = y - (1 * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = False
    
    pygame.display.flip()
    clock.tick(150)

  '''# Sending data to the server and writing it on the respective file
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
  time.sleep(0.3)
  casual_data = s.recv(1024)
  time.sleep(0.3)
  file1.write(bytes.decode(casual_data))
  file1.write("\n\n VerticalMove END \n\n")
  file1.close()
  time.sleep(0.3)'''


#Box text that moves in diagonal
def diagMove():
 
  diag = math.sqrt((sizeWidth*sizeWidth)+(sizeHeight*sizeHeight))

  #Starting image position
  x = 0
  y = 0
  speed = 0.2

  #Load and rescale the text image
  #picture = pygame.image.load('movingText/Images/text3.png').convert()
  picture = pygame.image.load('Programs/Images/text3.png').convert()
  picture = pygame.transform.scale(picture, (300, 150))

  '''# File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\Davide Mascheroni\\Desktop\\Risultati\\Results{}-Trial{}.txt".format(testernumber, index), "a")'''

  # Setting the time
  test_time = 10
  t_end = time.time() + test_time

  wall = False

  while time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()


    while not wall and time.time() <= t_end:

      '''# Sending data to the server and writing it on the respective file
      casual_data = s.recv(1024)
      file1.write(bytes.decode(casual_data))'''

      if (x < sizeWidth - 300):
        x = x + (1*(sizeWidth/diag) * speed)
        y = y + (1 *(sizeHeight/diag) * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = True

    while wall and time.time() <= t_end:

      '''# Sending data to the server and writing it on the respective file
      casual_data = s.recv(1024)
      file1.write(bytes.decode(casual_data))
      '''
      if (x > 1):
        x = x - (1*(sizeWidth/diag) * speed)
        y = y - (1 *(sizeHeight/diag) * speed)
        screen.fill(BLACK)
        screen.blit(picture, (x,y))
        pygame.display.flip()
      else:
        wall = False

     
    pygame.display.flip()
    clock.tick(150)

  '''# Sending data to the server and writing it on the respective file
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
  time.sleep(0.3)
  casual_data = s.recv(1024)
  time.sleep(0.3)
  file1.write(bytes.decode(casual_data))
  file1.write("\n\n DiagonalMove END \n\n")
  file1.close()
  time.sleep(0.3)'''



#Text scroll from right to left
def horizontalScroll():

  dim_char = 45
  #Create a font
  font = pygame.font.SysFont("Arial", dim_char)
  #Text to show
  text = "Milan is a city in northern Italy, regional capital of Lombardy, the largest city in Italy by urban population and the second-most-populous city proper in Italy after Rome. The city proper has a population of about 1.4 million, while its metropolitan city has 3.25 million residents. The urban area of Milan is the fourth-most-populous in the EU with 6.17 million inhabitants. According to national sources, the population within the wider Milan metropolitan area is estimated between 7.5 million and 8.2 million, making it by far the largest metropolitan area in Italy and one of the largest in the EU. Milan is the economic capital of Italy, one of the economic capitals of Europe and a global financial centre."
  #Get text width and height
  text_width, text_height = font.size(text)
  # Setting the time
  test_time = 10
  t_end = time.time() + test_time

  #Starting image position and speed
  x = sizeWidth
  y = (sizeHeight / 2) - (text_height / 2)
  speed = 1.4

  '''# File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\Davide Mascheroni\\Desktop\\Risultati\\Results{}-Trial{}.txt".format(testernumber, index), "a")'''

  while (x > -text_width) and time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()

    '''# Sending data to the server and writing it on the respective file
    casual_data = s.recv(1024)
    file1.write(bytes.decode(casual_data))'''

    x = x - (1*speed)
    screen.fill(BLACK)
    img = font.render(text, True, WHITE)
    screen.blit(img, (x, y))
  
    pygame.display.flip()
    clock.tick(150)

  '''# Sending data to the server and writing it on the respective file
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
  time.sleep(0.3)
  casual_data = s.recv(1024)
  time.sleep(0.3)
  file1.write(bytes.decode(casual_data))
  file1.write("\n\n Horizontalscroll END \n\n")
  file1.close()
  time.sleep(0.3)'''



#Block of text that moves vertically
def verticalBlock():

  dim_char = 45
  #Create a font
  font = pygame.font.SysFont("Arial", 45)
  #Text to show
  text = "Eclipses only occur when the Sun, Earth, and Moon are all in a straight line. Solar eclipses occur at new moon, when the Moon is between the Sun and Earth. In contrast, lunar eclipses occur at full moon, when Earth is between the Sun and Moon. The apparent size of the Moon is roughly the same as that of the Sun, with both being viewed at close to one-half a degree wide. The Sun is much larger than the Moon, but it is the vastly greater distance that gives it the same apparent size as the much closer and much smaller Moon from the perspective of Earth. The variations in apparent size, due to the non-circular orbits, are nearly the same as well, though occurring in different cycles. This makes possible both total (with the Moon appearing larger than the Sun) and annular (with the Moon appearing smaller than the Sun) solar eclipses.[217] In a total eclipse, the Moon completely covers the disc of the Sun and the solar corona becomes visible to the naked eye."
  #Get text width and height
  text_width, text_height = font.size(text)
  # Setting the time
  test_time = 10
  #Number of characters per line
  n = 30
  t_end = time.time() + test_time

  #Starting image position and speed
  x = sizeWidth/2 - (text_width/(n*2))
  y = sizeHeight
  speed = 0.5
  text = addSeparator(text, n)
  #text with new line
  nlText = text.split("#")

  '''# File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\Davide Mascheroni\\Desktop\\Risultati\\Results{}-Trial{}.txt".format(testernumber, index), "a")'''

  while (x > -text_width) and time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()
    
    '''# Sending data to the server and writing it on the respective file
    casual_data = s.recv(1024)
    file1.write(bytes.decode(casual_data))'''

    y = y - (1*speed)
    screen.fill(BLACK)
    createVertBlock(x, y, font, nlText, 45)
  
    pygame.display.flip()
    clock.tick(150)

  '''# Sending data to the server and writing it on the respective file
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
  time.sleep(0.3)
  casual_data = s.recv(1024)
  time.sleep(0.3)
  file1.write(bytes.decode(casual_data))
  file1.write("\n\n VerticalBlock END \n\n")
  file1.close()
  time.sleep(0.3)'''


def main():
    
  pygame.init()
  pygame.mouse.set_visible(False)

  '''#shuffle the order of the animations
  tests_list = [horizontalMove, verticalMove, diagMove, horizontalScroll, verticalBlock]
  random.shuffle(tests_list)

  #run the animation after the shuffle
  for funct in tests_list:
    funct()'''
  
  horizontalMove()    
  
  pygame.quit()
  '''s.close()'''

if __name__ == "__main__":
    main()


