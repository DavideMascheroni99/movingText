import pygame
import tkinter
from tkinter import simpledialog
import os
import socket
import sys
import time
import random
import mysql.connector
import glb_var_const
import exception
import datetime
from pathlib import Path



# Dialogue window for the tester number
application_window = tkinter.Tk()
#Ask the tester number
tester_number = simpledialog.askstring("Input", "Input tester_number", parent=application_window)

#Set environment variables
os.environ['SDL_VIDEO_WINDOW_POS'] = '{0},{1}'.format(glb_var_const.win_pos_left, glb_var_const.win_pos_top)
#initialize display for a window
screen = pygame.display.set_mode(glb_var_const.winsize, pygame.FULLSCREEN)
pygame.display.set_caption("Test")

# Used to manage how fast the screen updates. Create an object to manage the time
clock = pygame.time.Clock()

# Wrap text into lines that fit a given width, no leading spaces
def wrap_text(text, font, max_width):
  words = text.split(' ')
  lines = []
  current_line = ''
  for word in words:
    test_line = current_line + word + ' '
    if font.size(test_line)[0] <= max_width:
      current_line = test_line
    else:
      if current_line:  # Ensure non-empty lines
        lines.append(current_line.strip())  # Remove leading/trailing spaces
      current_line = word + ' '  # Start a new line with the word
  if current_line:  # Append the last line
    lines.append(current_line.strip())
  return lines


# Render all lines into one surface
def render_text_surface(text, font, color, width, line_spacing=6):
  lines = wrap_text(text, font, width)
  line_height = font.get_height()
  total_height = len(lines) * line_height + (len(lines) - 1) * line_spacing

  surface = pygame.Surface((width, total_height), pygame.SRCALPHA)
  surface = surface.convert_alpha()

  y = 0
  for line in lines:
    rendered = font.render(line, True, color)
    surface.blit(rendered, (0, y))  # Align text to the left (no padding)
    y += line_height + line_spacing

  return surface

  
#Create a white cross to display
def draw_fixation_cross(x, y, length=20, width=5, color=pygame.Color(glb_var_const.WHITE)):
  pygame.draw.line(screen, color, (x, y - length), (x, y + length), width)
  pygame.draw.line(screen, color, (x - length, y), (x + length, y), width)


#Show white cross for tcross seconds
def show_white_cross():
  t_end = time.time() + glb_var_const.TCROSS
  while time.time() <= t_end:
    screen.fill(pygame.Color(glb_var_const.BLACK)) 
    draw_fixation_cross(glb_var_const.center_x, glb_var_const.center_y)
    pygame.display.flip()  


def horizontal_scroll(text, speed, dim_char, fname):
    show_white_cross()
    font = pygame.font.SysFont(glb_var_const.FONT, dim_char)
    text_width, text_height = font.size(text)
    t_end = time.time() + glb_var_const.TEST_TIME

    x = glb_var_const.screen_width
    y = (glb_var_const.screen_height / 2) - (text_height / 2)

    while time.time() <= t_end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit()
                if event.key == pygame.K_a:
                    return "restart"

        x -= speed

        screen.fill(glb_var_const.BLACK)
        img = font.render(text, True, glb_var_const.WHITE)
        screen.blit(img, (x, y))

        pygame.display.flip()
        clock.tick(150)

    return "done"


def hor_scroll_big():
  fname = "HS-BIG"
  speed = glb_var_const.STARTING_SPEED_HS
  while True:
    text = random.choice(glb_var_const.allTexts)
    result = horizontal_scroll(text, speed, glb_var_const.BIG_CHAR, fname)

    if result == "done":
      # If the time runs out (after 10 seconds), save the previous speed - 0.1
      with open("movingText\\Files\\speed_log.txt", "a") as file:
        file.write("{} : {} \n".format(fname, speed-glb_var_const.FACTOR))

    if result == "restart":
         # Increase speed by 0.1 when restarted
        speed += glb_var_const.FACTOR
        # Restart the scroll
        continue
    else:
        # Exit after 10 seconds
        break 
  

def hor_scroll_little():
  fname = "HS-LITTLE"
  speed = glb_var_const.STARTING_SPEED_HS
  while True:
    text = random.choice(glb_var_const.allTexts)
    result = horizontal_scroll(text, speed, glb_var_const.LITTLE_CHAR, fname)

    if result == "done":
      # If the time runs out (after 10 seconds), save the previous speed - 0.1
      with open("movingText\\Files\\speed_log.txt", "a") as file:
        file.write("{} : {} \n".format(fname, speed-glb_var_const.FACTOR))

    if result == "restart":
         # Increase speed by 0.1 when restarted
        speed += glb_var_const.FACTOR
        # Restart the scroll
        continue
    else:
        # Exit after 10 seconds
        break   
    

#Block of text that moves vertically
def vertical_block(text, speed, dim_char, fname):
  show_white_cross()
  t_end = time.time() + glb_var_const.TEST_TIME
  #Create a font
  font = pygame.font.SysFont(glb_var_const.FONT, dim_char)
  
  text_width = glb_var_const.screen_width * 2 // 3
  text_surface = render_text_surface(text, font, glb_var_const.WHITE, text_width)
  text_rect = text_surface.get_rect(centerx=glb_var_const.screen_width // 2)
  y_pos = float(glb_var_const.screen_height)
  text_rect.y = int(y_pos)


  while time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          sys.exit()
        if event.key == pygame.K_a:
          return "restart"
    
    y_pos -= speed
    text_rect.y = int(y_pos)

    screen.fill(glb_var_const.BLACK)
    screen.blit(text_surface, text_rect)
    pygame.display.flip()
    clock.tick(150)
  return "done"


def vert_block_big():
  fname = "VB-BIG"
  speed = glb_var_const.STARTING_SPEED_VB
  while True:
    text = random.choice(glb_var_const.allTexts)
    result = vertical_block(text, speed, glb_var_const.BIG_CHAR, fname)

    if result == "done":
      # If the time runs out (after 10 seconds), save the previous speed - 0.1
      with open("movingText\\Files\\speed_log.txt", "a") as file:
        file.write("{} : {} \n".format(fname, speed-glb_var_const.FACTOR))

    if result == "restart":
         # Increase speed by 0.1 when restarted
        speed += glb_var_const.FACTOR
        # Restart the scroll
        continue
    else:
        # Exit after 10 seconds
        break 
    

def vert_block_little():
  fname = "VB-LITTLE"
  speed = glb_var_const.STARTING_SPEED_VB
  while True:
    text = random.choice(glb_var_const.allTexts)
    result = vertical_block(text, speed, glb_var_const.LITTLE_CHAR, fname)

    if result == "done":
      # If the time runs out (after 10 seconds), save the previous speed - 0.1
      with open("movingText\\Files\\speed_log.txt", "a") as file:
        file.write("{} : {} \n".format(fname, speed-glb_var_const.FACTOR))

    if result == "restart":
         # Increase speed by 0.1 when restarted
        speed += glb_var_const.FACTOR
        # Restart the scroll
        continue
    else:
        # Exit after 10 seconds
        break   
  


def main():
    
  pygame.init()
  pygame.mouse.set_visible(False)

  with open("movingText\\Files\\speed_log.txt", "a") as file:
      file.write("Tester : {}\n".format(tester_number))

  tests_list = [hor_scroll_big, hor_scroll_little, vert_block_big, vert_block_little]
  random.shuffle(tests_list)

  for function in tests_list:
     function()

  pygame.quit()
  '''s.close()'''

if __name__ == "__main__":
    main()

