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
#Ask the session number
session_number = simpledialog.askstring("Input", "Input session_number", parent=application_window)
#Insert the trial number
trial_number = simpledialog.askstring("Input", "Input trial_number", parent=application_window)

#Set environment variables
os.environ['SDL_VIDEO_WINDOW_POS'] = '{0},{1}'.format(glb_var_const.win_pos_left, glb_var_const.win_pos_top)
#initialize display for a window
screen = pygame.display.set_mode(glb_var_const.winsize, pygame.FULLSCREEN)
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
def show_white_cross(fname, type):
  Path("movingText\\Cross_Results\\Tester{}".format(tester_number)).mkdir(parents=True, exist_ok=True)
  Path("movingText\\Cross_Results\\Tester{}\\Session{}".format(tester_number, session_number)).mkdir(parents=True, exist_ok=True)
  Path("movingText\\Cross_Results\\Tester{}\\Session{}\\Trial{}".format(tester_number, session_number, trial_number)).mkdir(parents=True, exist_ok=True)

  # File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("movingText\\Cross_Results\\Tester{}\\Session{}\\Trial{}\\T{}-S{}-TRY{}-{}_{}.txt".format(tester_number, session_number, trial_number, tester_number, session_number, trial_number, type, fname), "w")
  file1.write(str(datetime.datetime.now())+"\n")

  t_end = time.time() + glb_var_const.TCROSS
  while time.time() <= t_end:

    # Sending data to the server and writing it on the respective file
    casual_data = s.recv(1024)
    file1.write(bytes.decode(casual_data))

    screen.fill(pygame.Color(glb_var_const.BLACK)) 
    draw_fixation_cross(glb_var_const.center_x, glb_var_const.center_y)
    pygame.display.flip() 

  # Sending data to the server and writing it on the respective file
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
  time.sleep(0.3)
  casual_data = s.recv(1024)
  time.sleep(0.3)
  file1.write(bytes.decode(casual_data))
  file1.close()
  time.sleep(0.3) 


#Connect to the database
def db_connection():
  cnx = mysql.connector.connect(user='root', password='Dadeinter99', host='127.0.0.1', database='indexes')
  return cnx


#Used to check if the data are correctly collected in the database
def check_write_order(myresult):
  if(len(myresult) == 0): 
    raise exception.MyException("Wrong write order")


#Insert K generated texts in the database
def insert_k_texts(mycursor, text, conn):
  for i, j in zip(range(0,8), text):
    mycursor.execute("INSERT INTO rem_index (tester_number, session_number, trial_number, index_funct, txt) VALUES (%s, %s, %s, %s, %s)", (tester_number, session_number, trial_number, i, j))
    conn.commit()


#Generate k random texts
def gen_random_text(allTexts):
  text = random.sample(allTexts, glb_var_const.K)
  return text


#Remove previously used texts from the text list
def rem_used_texts(myresult):
  for j in range(len(myresult)):
    glb_var_const.allTexts.remove(myresult[j][0])


#Return fresh random text
def random_text():
  testerN_int = int(tester_number)
  sessionN_int = int(session_number)
  trialN_int = int(trial_number)
  #start the db connection
  conn = db_connection()
  mycursor = conn.cursor()
  
  #Check the correct input
  if((sessionN_int > 3 or sessionN_int < 1) or (trialN_int > 3 or trialN_int < 1) or (testerN_int < 1)):
      raise exception.MyException("Invalid input number")

  if(sessionN_int == 1 and trialN_int == 1):
    text = gen_random_text(glb_var_const.allTexts)
    #Execute the query
    mycursor.execute("SELECT * FROM rem_index WHERE tester_number = %s", (tester_number, ))
    row = mycursor.fetchone()
    if (row == None):
      insert_k_texts(mycursor, text, conn)
    else:
      raise exception.MyException("There are already data for tester {} during session {} and trial {} in the database".format(tester_number, session_number, trial_number))
    
  if(sessionN_int == 1 and trialN_int != 1):
    #Remove already used texts in previous trials
    for i in range (1, trialN_int):
      mycursor.execute("SELECT txt FROM rem_index WHERE tester_number = %s and trial_number = %s", (tester_number, i))
      myresult = mycursor.fetchall()
      check_write_order(myresult)
      rem_used_texts(myresult)

    text = gen_random_text(glb_var_const.allTexts)
    insert_k_texts(mycursor, text, conn)

  if(sessionN_int !=1 and trialN_int == 1):
    
    #Check correct order
    for i in range(1, sessionN_int):
      for j in range(1, 4):
        mycursor.execute("SELECT txt FROM rem_index WHERE tester_number = %s and session_number = %s and trial_number = %s", (tester_number, i, str(j)))
        myresult = mycursor.fetchall()
        check_write_order(myresult)

    #Remove already used texts in previous sessions
    for i in range(1, sessionN_int):
      mycursor.execute("SELECT txt FROM rem_index WHERE tester_number = %s and session_number = %s", (tester_number, i))
      myresult = mycursor.fetchall()
      rem_used_texts(myresult)
    text = gen_random_text(glb_var_const.allTexts)
    insert_k_texts(mycursor, text, conn)

  if(sessionN_int !=1 and trialN_int != 1):
    #Remove already used texts in previous sessions
    for i in range(1, sessionN_int):
      mycursor.execute("SELECT txt FROM rem_index WHERE tester_number = %s and session_number = %s", (tester_number, i))
      myresult = mycursor.fetchall()
      check_write_order(myresult)
      rem_used_texts(myresult)
    #Remove already used texts in the previous trials
    for i in range(1, trialN_int):
      mycursor.execute("SELECT txt FROM rem_index WHERE tester_number = %s and trial_number = %s and session_number = %s", (tester_number, i, session_number))
      myresult = mycursor.fetchall()
      check_write_order(myresult)
      rem_used_texts(myresult)
    text = gen_random_text(glb_var_const.allTexts)
    insert_k_texts(mycursor, text, conn)

  #close the db connection
  conn.close()
  return text


#Text scroll from right to left
def horizontal_scroll(text, speed, dim_char, fname):
  show_white_cross(fname, "HS")
  #Create a font
  font = pygame.font.SysFont(glb_var_const.FONT, dim_char)
  #Get text width and height
  text_width, text_height = font.size(text)
  t_end = time.time() + glb_var_const.TEST_TIME
  #Starting image position and speed
  x = glb_var_const.screen_width
  y = (glb_var_const.screen_height / 2) - (text_height / 2)

  Path("movingText\\Results\\Tester{}".format(tester_number)).mkdir(parents=True, exist_ok=True)
  Path("movingText\\Results\\Tester{}\\Session{}".format(tester_number, session_number)).mkdir(parents=True, exist_ok=True)
  Path("movingText\\Results\\Tester{}\\Session{}\\Trial{}".format(tester_number, session_number, trial_number)).mkdir(parents=True, exist_ok=True)

  # File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("movingText\\Results\\Tester{}\\Session{}\\Trial{}\\T{}-S{}-TRY{}-HS_{}.txt".format(tester_number, session_number, trial_number, tester_number, session_number, trial_number, fname), "w")
  file1.write(str(datetime.datetime.now())+"\n")

  while time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()

    # Sending data to the server and writing it on the respective file
    casual_data = s.recv(1024)
    file1.write(bytes.decode(casual_data))

    x = x - speed
  
    screen.fill(glb_var_const.BLACK)
    img = font.render(text, True, glb_var_const.WHITE)
    screen.blit(img, (x, y))
  
    pygame.display.flip()
    clock.tick(150)

  # Sending data to the server and writing it on the respective file
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
  time.sleep(0.3)
  casual_data = s.recv(1024)
  time.sleep(0.3)
  file1.write(bytes.decode(casual_data))
  file1.close()
  time.sleep(0.3)


#Block of text that moves vertically
def vertical_block(text, speed, dim_char, fname):
  show_white_cross(fname, "VB")
  t_end = time.time() + glb_var_const.TEST_TIME
  #Create a font
  font = pygame.font.SysFont(glb_var_const.FONT, dim_char)
  
  text_width = glb_var_const.screen_width * 2 // 3
  text_surface = render_text_surface(text, font, glb_var_const.WHITE, text_width)
  text_rect = text_surface.get_rect(centerx=glb_var_const.screen_width // 2)
  
  y_pos = float(glb_var_const.screen_height)
  text_rect.y = int(y_pos)

  Path("movingText\\Results\\Tester{}".format(tester_number)).mkdir(parents=True, exist_ok=True)
  Path("movingText\\Results\\Tester{}\\Session{}".format(tester_number, session_number)).mkdir(parents=True, exist_ok=True)
  Path("movingText\\Results\\Tester{}\\Session{}\\Trial{}".format(tester_number, session_number, trial_number)).mkdir(parents=True, exist_ok=True)
 
  # File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("movingText\\Results\\Tester{}\\Session{}\\Trial{}\\T{}-S{}-TRY{}-VB_{}.txt".format(tester_number, session_number, trial_number, tester_number, session_number, trial_number, fname), "w")
  file1.write(str(datetime.datetime.now())+"\n")
  
  while time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()
 
    # Sending data to the server and writing it on the respective file
    casual_data = s.recv(1024)
    file1.write(bytes.decode(casual_data))
    
    y_pos -= speed
    text_rect.y = int(y_pos)

    screen.fill(glb_var_const.BLACK)
    screen.blit(text_surface, text_rect)
    pygame.display.flip()
    clock.tick(150)
  

  # Sending data to the server and writing it on the respective file
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
  time.sleep(0.3)
  casual_data = s.recv(1024)
  time.sleep(0.3)
  file1.write(bytes.decode(casual_data))
  file1.close()
  time.sleep(0.3)


def hor_scroll_slow_little(txt):
  horizontal_scroll(txt, glb_var_const.LOW_SPEED_HS_LIT, glb_var_const.LITTLE_CHAR, "SL_LIT")

def hor_scroll_slow_big(txt):
  horizontal_scroll(txt, glb_var_const.LOW_SPEED_HS_BIG, glb_var_const.BIG_CHAR, "SL_BIG")

def hor_scroll_fast_little(txt):
  horizontal_scroll(txt, glb_var_const.HIGH_SPEED_HS_LIT, glb_var_const.LITTLE_CHAR, "FA_LIT")

def hor_scroll_fast_big(txt):
  horizontal_scroll(txt, glb_var_const.HIGH_SPEED_HS_BIG, glb_var_const.BIG_CHAR, "FA_BIG")

def vert_block_slow_little(txt):
  vertical_block(txt, glb_var_const.LOW_SPEED_VB_LIT, glb_var_const.LITTLE_CHAR, "SL_LIT")

def vert_block_slow_big(txt):
  vertical_block(txt, glb_var_const.LOW_SPEED_VB_BIG, glb_var_const.BIG_CHAR, "SL_BIG")

def vert_block_fast_little(txt):
  vertical_block(txt, glb_var_const.HIGH_SPEED_VB_LIT, glb_var_const.LITTLE_CHAR, "FA_LIT")

def vert_block_fast_big(txt):
  vertical_block(txt, glb_var_const.HIGH_SPEED_VB_BIG, glb_var_const.BIG_CHAR, "FA_BIG")


def main():
    
  pygame.init()
  pygame.mouse.set_visible(False)

  #shuffle the order of the animations
  tests_list = [hor_scroll_slow_big, hor_scroll_slow_little, hor_scroll_fast_big, hor_scroll_fast_little, vert_block_slow_little, vert_block_slow_big, vert_block_fast_little, vert_block_fast_big]
  random.shuffle(tests_list)

  text = random_text()

  #run the animation after the shuffle
  for funct, txt in zip(tests_list, text):
    funct(txt)
  
  
  pygame.quit()
  s.close()

if __name__ == "__main__":
    main()

