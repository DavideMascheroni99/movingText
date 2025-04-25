import os
import random
import sys
import socket
import pygame
import math
import time
import _random
import tkinter
from tkinter import simpledialog


'''GLOBAL VARIABLES AND CONSTANTS'''

# Dialogue window for the tester number
application_window = tkinter.Tk()
testernumber = simpledialog.askstring("Input", "Input tester number", parent=application_window)

# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Window creation
winsize = (width, height) = [1920, 1080]
center_x = winsize[0] // 2
center_y = winsize[1] // 2
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


'''CLASSES AND FUNCTIONS'''

class Rect:

    def __init__(self):
        self.x = 0
        self.y = 0
        self.dx = 0
        self.dy = 0
        self.size = 50
        self.speed = 0
        self.color = WHITE

# Synchronizaion screen
def synch(tester_number, i, duration):
    dim_1 = 70
    dim_2 = 10
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))

    # File to write the data on
    file1 = open("C:\\Users\\Roberto Gentilini\\Desktop\\risultati_txt\\{}synch{}.txt".format(tester_number, i), "w")

    t_end = time.time() + duration
    while time.time() <= t_end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()

        # Sending data to the server and writing it on the respective file
        synch_data = s.recv(1024)
        file1.write(bytes.decode(synch_data))
 
        screen.fill(BLACK)
        pygame.draw.rect(screen, WHITE, [width / 2 - dim_1 / 2, height / 2 - dim_2 / 2, dim_1, dim_2])
        pygame.draw.rect(screen, WHITE, [width / 2 - dim_2 / 2, height / 2 - dim_1 / 2, dim_2, dim_1])
        pygame.display.flip()

        # Frames limitation to 150 per second
        clock.tick(150)

    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
    time.sleep(0.3)
    synch_data = s.recv(1024)
    time.sleep(0.3)
    file1.write(bytes.decode(synch_data))

    file1.close()

    pass

# Square drawer
def draw_function(window, color, pos_x, pos_y, size_x, size_y):
    pygame.draw.rect(window, color, [pos_x, pos_y, size_x, size_y])
    pass

# Converter from degrees to sine/cosine
def convert_angle(ang_deg):
    ang_rad = math.radians(ang_deg)
    cos_x = math.cos(ang_rad)
    sin_y = math.sin(ang_rad)
    return cos_x, sin_y

# Wall bounce checker
def wall_bounce(x, y, dx, dy, size_x, size_y, width_screen, height_screen):
    if x > (width_screen - size_x) or x < 0:
        dx = dx * -1
    if y > (height_screen - size_y) or y < 0:
        dy = dy * -1
    return dx, dy

# Position updater
def update_pos(x, y, dx, dy, speed):
    x += dx * speed
    y += dy * speed
    return x, y


'''FUNCTIONS USED DURING THE TEST'''

def test_vertical_single(tester_number, testindex):
    rect1 = Rect()
    rect1.x = (winsize[0] // 2) - (rect1.size // 2)
    rect1.y = (winsize[1] // 2) - (rect1.size // 2)
    rect1.speed = 2
    rect1.dx, rect1.dy = convert_angle(90)
    rect1.dx *= rect1.speed
    rect1.dy *= rect1.speed
    
    rect2 = Rect()
    rect2.x = rect1.x
    rect2.y = rect1.y
    rect2.speed = 2
    rect2.dx, rect2.dy = convert_angle(270)
    rect2.dx *= rect2.speed
    rect2.dy *= rect2.speed

    # File to write on
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
    file1 = open("C:\\Users\\Roberto Gentilini\\Desktop\\risultati_txt\\{}test{}verticalsing.txt".format(tester_number, testindex), "w")
    # Setting the time
    test_time = 9
    t_end = time.time() + test_time

    while time.time() <= t_end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()

        # Sending data to the server and writing it on the respective file
        casual_data = s.recv(1024)
        file1.write(bytes.decode(casual_data))

        # Movement via object repositioning
        rect1.x, rect1.y = update_pos(rect1.x, rect1.y, rect1.dx, rect1.dy, rect1.speed)
        rect2.x, rect2.y = update_pos(rect2.x, rect2.y, rect2.dx, rect2.dy, rect2.speed)
        screen.fill(BLACK)
        draw_function(screen, WHITE, rect1.x, rect1.y, rect1.size, rect1.size)
        draw_function(screen, WHITE, rect2.x, rect2.y, rect2.size, rect2.size)
        rect1.dx, rect1.dy = wall_bounce(rect1.x, rect1.y, rect1.dx, rect1.dy, rect1.size, rect1.size, width, height)
        rect2.dx, rect2.dy = wall_bounce(rect2.x, rect2.y, rect2.dx, rect2.dy, rect2.size, rect2.size, width, height)
        
        pygame.display.flip()
        clock.tick(150)

    # Sending data to the server and writing it on the respective file
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
    time.sleep(0.3)
    casual_data = s.recv(1024)
    time.sleep(0.3)
    file1.write(bytes.decode(casual_data))
    file1.close()

    pass


def test_horizontal_single(tester_number, testindex):
    rect1 = Rect()
    rect1.x = (winsize[0] // 2) - (rect1.size // 2)
    rect1.y = (winsize[1] // 2) - (rect1.size // 2)
    rect1.speed = 2.5
    rect1.dx, rect1.dy = convert_angle(0)
    rect1.dx *= rect1.speed
    rect1.dy *= rect1.speed
    
    rect2 = Rect()
    rect2.x = rect1.x
    rect2.y = rect1.y
    rect2.speed = 2.5
    rect2.dx, rect2.dy = convert_angle(180)
    rect2.dx *= rect2.speed
    rect2.dy *= rect2.speed

    # File to write on
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
    file1 = open("C:\\Users\\Roberto Gentilini\\Desktop\\risultati_txt\\{}test{}horizontalsing.txt".format(tester_number, testindex), "w")
    # Setting the time
    test_time = 9
    t_end = time.time() + test_time

    while time.time() <= t_end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()

        # Sending data to the server and writing it on the respective file
        casual_data = s.recv(1024)
        file1.write(bytes.decode(casual_data))

        # Movement via object repositioning
        rect1.x, rect1.y = update_pos(rect1.x, rect1.y, rect1.dx, rect1.dy, rect1.speed)
        rect2.x, rect2.y = update_pos(rect2.x, rect2.y, rect2.dx, rect2.dy, rect2.speed)
        screen.fill(BLACK)
        draw_function(screen, WHITE, rect1.x, rect1.y, rect1.size, rect1.size)
        draw_function(screen, WHITE, rect2.x, rect2.y, rect2.size, rect2.size)
        rect1.dx, rect1.dy = wall_bounce(rect1.x, rect1.y, rect1.dx, rect1.dy, rect1.size, rect1.size, width, height)
        rect2.dx, rect2.dy = wall_bounce(rect2.x, rect2.y, rect2.dx, rect2.dy, rect2.size, rect2.size, width, height)
        
        pygame.display.flip()
        clock.tick(150)

    # Sending data to the server and writing it on the respective file
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
    time.sleep(0.3)
    casual_data = s.recv(1024)
    time.sleep(0.3)
    file1.write(bytes.decode(casual_data))
    file1.close()

    pass


def test_vertical(tester_number, testindex):

    rectnumber = 5
    rectdist = winsize[0] // (rectnumber)
    rect1 = []
    rect2 = []
    
    for i in range(rectnumber):
        
        rect1.append(Rect())
        rect1[i].x = (rectdist // 2) + (i * rectdist) - (rect1[i].size // 2)
        rect1[i].y = (winsize[1] // 2) - (rect1[i].size // 2)
        rect1[i].speed = 2
        rect1[i].dx, rect1[i].dy = convert_angle(90)
        rect1[i].dx *= rect1[i].speed
        rect1[i].dy *= rect1[i].speed
        
        rect2.append(Rect())
        rect2[i].x = rect1[i].x
        rect2[i].y = (winsize[1] // 2) - (rect1[i].size // 2)
        rect2[i].speed = 2
        rect2[i].dx, rect2[i].dy = convert_angle(270)
        rect2[i].dx *= rect2[i].speed
        rect2[i].dy *= rect2[i].speed
    
    # File to write on
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
    file1 = open("C:\\Users\\Roberto Gentilini\\Desktop\\risultati_txt\\{}test{}verticalmult.txt".format(tester_number, testindex), "w")
    # Setting the time
    test_time = 9
    t_end = time.time() + test_time

    while time.time() <= t_end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()
        
        # Sending data to the server and writing it on the respective file
        casual_data = s.recv(1024)
        file1.write(bytes.decode(casual_data))
        
        screen.fill(BLACK)
        
        # Movement via object repositioning
        for i in range(rectnumber):
            rect1[i].x, rect1[i].y = update_pos(rect1[i].x, rect1[i].y, rect1[i].dx, rect1[i].dy, rect1[i].speed)
            rect2[i].x, rect2[i].y = update_pos(rect2[i].x, rect2[i].y, rect2[i].dx, rect2[i].dy, rect2[i].speed)

            draw_function(screen, WHITE, rect1[i].x, rect1[i].y, rect1[i].size, rect1[i].size)
            draw_function(screen, WHITE, rect2[i].x, rect2[i].y, rect2[i].size, rect2[i].size)

            rect1[i].dx, rect1[i].dy = wall_bounce(rect1[i].x, rect1[i].y, rect1[i].dx, rect1[i].dy, rect1[i].size, rect1[i].size, width, height)
            rect2[i].dx, rect2[i].dy = wall_bounce(rect2[i].x, rect2[i].y, rect2[i].dx, rect2[i].dy, rect2[i].size, rect2[i].size, width, height)

        pygame.display.flip()
        clock.tick(150)
    
    # Sending data to the server and writing it on the respective file
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
    time.sleep(0.3)
    casual_data = s.recv(1024)
    time.sleep(0.3)
    file1.write(bytes.decode(casual_data))
    file1.close()
    
    pass


def test_horizontal(tester_number, testindex):

    rectnumber = 4
    rectdist = winsize[1] // (rectnumber)
    rect1 = []
    rect2 = []
    
    for i in range(rectnumber):
        
        rect1.append(Rect())
        rect1[i].x = (winsize[0] // 2) - (rect1[i].size // 2)
        rect1[i].y = (rectdist // 2) + (i * rectdist) - (rect1[i].size // 2)
        rect1[i].speed = 2.5
        rect1[i].dx, rect1[i].dy = convert_angle(180)
        rect1[i].dx *= rect1[i].speed
        rect1[i].dy *= rect1[i].speed
        
        rect2.append(Rect())
        rect2[i].x = (winsize[0] // 2) - (rect1[i].size // 2)
        rect2[i].y = rect1[i].y
        rect2[i].speed = 2.5
        rect2[i].dx, rect2[i].dy = convert_angle(0)
        rect2[i].dx *= rect2[i].speed
        rect2[i].dy *= rect2[i].speed

    # File to write on
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
    file1 = open("C:\\Users\\Roberto Gentilini\\Desktop\\risultati_txt\\{}test{}horizontalmult.txt".format(tester_number, testindex), "w")
    # Setting the time
    test_time = 9
    t_end = time.time() + test_time
    
    i = 0

    while time.time() <= t_end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()
        
        # Sending data to the server and writing it on the respective file
        casual_data = s.recv(1024)
        file1.write(bytes.decode(casual_data))

        screen.fill(BLACK)
        
        # Movement via object repositioning
        for i in range(rectnumber):
            rect1[i].x, rect1[i].y = update_pos(rect1[i].x, rect1[i].y, rect1[i].dx, rect1[i].dy, rect1[i].speed)
            rect2[i].x, rect2[i].y = update_pos(rect2[i].x, rect2[i].y, rect2[i].dx, rect2[i].dy, rect2[i].speed)

            draw_function(screen, WHITE, rect1[i].x, rect1[i].y, rect1[i].size, rect1[i].size)
            draw_function(screen, WHITE, rect2[i].x, rect2[i].y, rect2[i].size, rect2[i].size)

            rect1[i].dx, rect1[i].dy = wall_bounce(rect1[i].x, rect1[i].y, rect1[i].dx, rect1[i].dy, rect1[i].size, rect1[i].size, width, height)
            rect2[i].dx, rect2[i].dy = wall_bounce(rect2[i].x, rect2[i].y, rect2[i].dx, rect2[i].dy, rect2[i].size, rect2[i].size, width, height)

        pygame.display.flip()
        clock.tick(150)

    # Sending data to the server and writing it on the respective file
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
    time.sleep(0.3)
    casual_data = s.recv(1024)
    time.sleep(0.3)
    file1.write(bytes.decode(casual_data))
    file1.close()
    
    pass


def test_diagonal(tester_number, testindex):
    
    speed_diag = 2.5

    rect1 = Rect()
    rect1.x = (winsize[0] // 2) - (rect1.size // 2)
    rect1.y = (winsize[1] // 2) - (rect1.size // 2)
    rect1.speed = speed_diag
    rect1.dx, rect1.dy = convert_angle(28.9)
    rect1.dx *= rect1.speed
    rect1.dy *= rect1.speed

    rect2 = Rect()
    rect2.x = (winsize[0] // 2) - (rect2.size // 2)
    rect2.y = (winsize[1] // 2) - (rect2.size // 2)
    rect2.speed = speed_diag
    rect2.dx, rect2.dy = convert_angle(151.1)
    rect2.dx *= rect2.speed
    rect2.dy *= rect2.speed

    rect3 = Rect()
    rect3.x = (winsize[0] // 2) - (rect3.size // 2)
    rect3.y = (winsize[1] // 2) - (rect3.size // 2)
    rect3.speed = speed_diag
    rect3.dx, rect3.dy = convert_angle(208.9)
    rect3.dx *= rect3.speed
    rect3.dy *= rect3.speed

    rect4 = Rect()
    rect4.x = (winsize[0] // 2) - (rect4.size // 2)
    rect4.y = (winsize[1] // 2) - (rect4.size // 2)
    rect4.speed = speed_diag
    rect4.dx, rect4.dy = convert_angle(331.1)
    rect4.dx *= rect4.speed
    rect4.dy *= rect4.speed

    # File to write on
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
    file1 = open("C:\\Users\\Roberto Gentilini\\Desktop\\risultati_txt\\{}test{}diagonal.txt".format(tester_number, testindex), "w")
    # Setting the time
    test_time = 9
    t_end = time.time() + test_time

    while time.time() <= t_end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()
                
        # Sending data to the server and writing it on the respective file
        casual_data = s.recv(1024)
        file1.write(bytes.decode(casual_data))

        # Movement via object repositioning
        rect1.x, rect1.y = update_pos(rect1.x, rect1.y, rect1.dx, rect1.dy, rect1.speed)
        rect2.x, rect2.y = update_pos(rect2.x, rect2.y, rect2.dx, rect2.dy, rect2.speed)
        rect3.x, rect3.y = update_pos(rect3.x, rect3.y, rect3.dx, rect3.dy, rect3.speed)
        rect4.x, rect4.y = update_pos(rect4.x, rect4.y, rect4.dx, rect4.dy, rect4.speed)

        screen.fill(BLACK)

        draw_function(screen, WHITE, rect1.x, rect1.y, rect1.size, rect1.size)
        draw_function(screen, WHITE, rect2.x, rect2.y, rect2.size, rect2.size)
        draw_function(screen, WHITE, rect3.x, rect3.y, rect3.size, rect3.size)
        draw_function(screen, WHITE, rect4.x, rect4.y, rect4.size, rect4.size)

        rect1.dx, rect1.dy = wall_bounce(rect1.x, rect1.y, rect1.dx, rect1.dy, rect1.size, rect1.size, width, height)
        rect2.dx, rect2.dy = wall_bounce(rect2.x, rect2.y, rect2.dx, rect2.dy, rect2.size, rect2.size, width, height)
        rect3.dx, rect3.dy = wall_bounce(rect3.x, rect3.y, rect3.dx, rect3.dy, rect3.size, rect3.size, width, height)
        rect4.dx, rect4.dy = wall_bounce(rect4.x, rect4.y, rect4.dx, rect4.dy, rect4.size, rect4.size, width, height)

        pygame.display.flip()
        clock.tick(150)

    # Sending data to the server and writing it on the respective file
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
    time.sleep(0.3)
    casual_data = s.recv(1024)
    time.sleep(0.3)
    file1.write(bytes.decode(casual_data))
    file1.close()

    pass
    
    
def test_radial(tester_number, testindex):
    
    speed_diag = 2.5
    speed_hor = 2.345
    speed_ver = 1.737

    rect1 = Rect()
    rect1.x = (winsize[0] // 2) - (rect1.size // 2)
    rect1.y = (winsize[1] // 2) - (rect1.size // 2)
    rect1.speed = speed_diag
    rect1.dx, rect1.dy = convert_angle(28.9)
    rect1.dx *= rect1.speed
    rect1.dy *= rect1.speed

    rect2 = Rect()
    rect2.x = (winsize[0] // 2) - (rect2.size // 2)
    rect2.y = (winsize[1] // 2) - (rect2.size // 2)
    rect2.speed = speed_diag
    rect2.dx, rect2.dy = convert_angle(151.1)
    rect2.dx *= rect2.speed
    rect2.dy *= rect2.speed

    rect3 = Rect()
    rect3.x = (winsize[0] // 2) - (rect3.size // 2)
    rect3.y = (winsize[1] // 2) - (rect3.size // 2)
    rect3.speed = speed_diag
    rect3.dx, rect3.dy = convert_angle(208.9)
    rect3.dx *= rect3.speed
    rect3.dy *= rect3.speed

    rect4 = Rect()
    rect4.x = (winsize[0] // 2) - (rect4.size // 2)
    rect4.y = (winsize[1] // 2) - (rect4.size // 2)
    rect4.speed = speed_diag
    rect4.dx, rect4.dy = convert_angle(331.1)
    rect4.dx *= rect4.speed
    rect4.dy *= rect4.speed
    
    rect5 = Rect()
    rect5.x = (winsize[0] // 2) - (rect5.size // 2)
    rect5.y = (winsize[1] // 2) - (rect5.size // 2)
    rect5.speed = speed_hor
    rect5.dx, rect5.dy = convert_angle(180)
    rect5.dx *= rect5.speed
    rect5.dy *= rect5.speed
    
    rect6 = Rect()
    rect6.x = (winsize[0] // 2) - (rect5.size // 2)
    rect6.y = rect5.y
    rect6.speed = speed_hor
    rect6.dx, rect6.dy = convert_angle(0)
    rect6.dx *= rect6.speed
    rect6.dy *= rect6.speed
    
    rect7 = Rect()
    rect7.x = (winsize[0] // 2) - (rect5.size // 2)
    rect7.y = (winsize[1] // 2) - (rect7.size // 2)
    rect7.speed = speed_ver
    rect7.dx, rect7.dy = convert_angle(90)
    rect7.dx *= rect7.speed
    rect7.dy *= rect7.speed
    
    rect8 = Rect()
    rect8.x = rect7.x
    rect8.y = (winsize[1] // 2) - (rect7.size // 2)
    rect8.speed = speed_ver
    rect8.dx, rect8.dy = convert_angle(270)
    rect8.dx *= rect8.speed
    rect8.dy *= rect8.speed

    # File to write on
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
    file1 = open("C:\\Users\\Roberto Gentilini\\Desktop\\risultati_txt\\{}test{}radial.txt".format(tester_number, testindex), "w")
    # Setting the time
    test_time = 9
    t_end = time.time() + test_time

    while time.time() <= t_end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                sys.exit()
                
        # Sending data to the server and writing it on the respective file
        casual_data = s.recv(1024)
        file1.write(bytes.decode(casual_data))

        # Movement via object repositioning
        rect1.x, rect1.y = update_pos(rect1.x, rect1.y, rect1.dx, rect1.dy, rect1.speed)
        rect2.x, rect2.y = update_pos(rect2.x, rect2.y, rect2.dx, rect2.dy, rect2.speed)
        rect3.x, rect3.y = update_pos(rect3.x, rect3.y, rect3.dx, rect3.dy, rect3.speed)
        rect4.x, rect4.y = update_pos(rect4.x, rect4.y, rect4.dx, rect4.dy, rect4.speed)
        rect5.x, rect5.y = update_pos(rect5.x, rect5.y, rect5.dx, rect5.dy, rect5.speed)
        rect6.x, rect6.y = update_pos(rect6.x, rect6.y, rect6.dx, rect6.dy, rect6.speed)
        rect7.x, rect7.y = update_pos(rect7.x, rect7.y, rect7.dx, rect7.dy, rect7.speed)
        rect8.x, rect8.y = update_pos(rect8.x, rect8.y, rect8.dx, rect8.dy, rect8.speed)

        screen.fill(BLACK)

        draw_function(screen, WHITE, rect1.x, rect1.y, rect1.size, rect1.size)
        draw_function(screen, WHITE, rect2.x, rect2.y, rect2.size, rect2.size)
        draw_function(screen, WHITE, rect3.x, rect3.y, rect3.size, rect3.size)
        draw_function(screen, WHITE, rect4.x, rect4.y, rect4.size, rect4.size)
        draw_function(screen, WHITE, rect5.x, rect5.y, rect5.size, rect5.size)
        draw_function(screen, WHITE, rect6.x, rect6.y, rect6.size, rect6.size)
        draw_function(screen, WHITE, rect7.x, rect7.y, rect7.size, rect7.size)
        draw_function(screen, WHITE, rect8.x, rect8.y, rect8.size, rect8.size)

        rect1.dx, rect1.dy = wall_bounce(rect1.x, rect1.y, rect1.dx, rect1.dy, rect1.size, rect1.size, width, height)
        rect2.dx, rect2.dy = wall_bounce(rect2.x, rect2.y, rect2.dx, rect2.dy, rect2.size, rect2.size, width, height)
        rect3.dx, rect3.dy = wall_bounce(rect3.x, rect3.y, rect3.dx, rect3.dy, rect3.size, rect3.size, width, height)
        rect4.dx, rect4.dy = wall_bounce(rect4.x, rect4.y, rect4.dx, rect4.dy, rect4.size, rect4.size, width, height)
        rect5.dx, rect5.dy = wall_bounce(rect5.x, rect5.y, rect5.dx, rect5.dy, rect5.size, rect5.size, width, height)
        rect6.dx, rect6.dy = wall_bounce(rect6.x, rect6.y, rect6.dx, rect6.dy, rect6.size, rect6.size, width, height)
        rect7.dx, rect7.dy = wall_bounce(rect7.x, rect7.y, rect7.dx, rect7.dy, rect7.size, rect7.size, width, height)
        rect8.dx, rect8.dy = wall_bounce(rect8.x, rect8.y, rect8.dx, rect8.dy, rect8.size, rect8.size, width, height)

        pygame.display.flip()
        clock.tick(150)

    # Sending data to the server and writing it on the respective file
    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
    time.sleep(0.3)
    casual_data = s.recv(1024)
    time.sleep(0.3)
    file1.write(bytes.decode(casual_data))
    file1.close()

    pass


'''MAIN'''

def main():

    # Initialization of the pygame modules
    pygame.init()

    # Invisible cursor
    pygame.mouse.set_visible(False)

    # The tests are shown in random order
    tests_list = [test_vertical_single, test_horizontal_single, test_vertical, test_horizontal, test_diagonal, test_radial]
    random.shuffle(tests_list)
    duration = 3

    for index, functi in enumerate(tests_list, start=1):
        if index > 1:
            duration = 2
        synch(testernumber, index, duration)
        functi(testernumber, index)
        pass

    # Closure of the pygame modules
    pygame.quit()

    # Closure of the connection
    s.close()

    pass


if __name__ == "__main__":
    main()
