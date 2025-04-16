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

#45 texts containing cities description
text1 = "Sesto San Giovanni, locally referred to as just Sesto, is a comune in the Metropolitan City of Milan, in the Italian region of Lombardy. It was awarded with the honorary title of città (city) by decree of 10 April 1954, signed by President Luigi Einaudi. An unimportant agglomerate of buildings until the 19th century, Sesto San Giovanni grew during the end of the 19th century and in the early 20th century, becoming the site of several industries, including companies such as Falck, Campari, Magneti Marelli and Breda. In that period the population increased rapidly, from 5,000 inhabitants in 1880 to 14,000 in 1911."
text2 = "Cinisello Balsamo is a comune of about 75,200 inhabitants in the Metropolitan City of Milan, in the Italian region of Lombardy, about 10 kilometres northeast of Milan city center. Cinisello Balsamo borders the following municipalities: Monza, Muggiò, Nova Milanese, Paderno Dugnano, Cusano Milanino, Sesto San Giovanni, Bresso. The current comune was formed in 1928 by the union of Cinisello and Balsamo, and received the honorary title of city through a presidential decree on 17 October 1972."
text3 = "Legnano is a town and comune municipality in the province of Milan, about 20 kilometres from central Milan. With 60,259, it is the thirteenth-most populous township in Lombardy. Legnano is located in the Alto Milanese and is crossed by the Olona River. The history of Legnano and its municipal area has been traced back to the 1st millennium BC via archaeological evidence. Already in remote times, in fact, the hills that line the Olona had proved to be habitable places. The town was established in 1261. Because of the historic victory of the Lombard League over Frederick Barbarossa at Legnano, it is the only town other than Rome named in the Italian national anthem. Every year the people of Legnano commemorate the battle with Palio di Legnano. In the institutional sphere, on 29 May, the date of the battle of Legnano, it was chosen as the regional holiday of Lombardy."
text4 = "Rho is a town and comune in the Metropolitan City of Milan in the Italian region of Lombardy, located about 14 kilometres northwest of Milan. The language spoken in Rho is Italian. Rho is lapped by the river Olona and crossed by its tributaries Bozzente and Lura, nowadays partially cloaked inside the town. At the north and east of the town, there is the road of national interest Strada statale 33 del Sempione, which in the past was crossing the town itself, in the current corso Europa. Rho is at the meeting point of railways linking Milan to Varese and Domodossola and Milan to Novara."
text5 = "Cologno Monzese is a comune in the Metropolitan City of Milan in the Italian region of Lombardy, located about 5 kilometres northeast of Milan. The population increased substantially after World War II, when many people from Southern Italy settled here. After being subject for a long time to the influence exerted by San Maurizio al Lambro Cologno received the honorary title of city with a presidential decree on September 19, 1996."
text6 = "Paderno Dugnano is a town and comune in the Metropolitan City of Milan, in Lombardy, northern Italy. It is bounded by comuni of Senago, Limbiate, Varedo, Cusano Milanino, Cormano, Nova Milanese, Bollate, Novate Milanese, Cinisello Balsamo. Paderno Dugnano is about 15 kilometers from the center of Milan. Following the unification of Italy in 1861, a reorganization of the internal subdivisions of the country ensued. By decree of 17 March 1869, the comuni of Paderno, Dugnano, Incirano, Cassina Amata, and Palazzolo Milanese were fused into a new comune called Paderno Milanese."
text7 = "Rozzano is a comune in the Metropolitan City of Milan, in the Italian region Lombardy, located about 9 kilometres south of Milan. Rozzano borders the following municipalities: Milan, Assago, Zibido San Giacomo, Opera, Pieve Emanuele, Basiglio. Rozzano received the honorary title of city with a presidential decree on July 21, 2003. The first government to officialise the birth of the municipality of Rozzano was that of Napoleon who in 1809 decreed the annexation of Torriggio and in 1811 those of Cassino Scanasio, Pontesesto and Quinto de' Stampi. The Austrians first annulled everything in 1816, but then reconsidered regarding Torriggio and Cassino Scanasio in 1841, while it was Vittorio Emanuele II in 1870 who sanctioned the final union with Pontesesto, which also brought Quinto as a dowry."
text8 = "San Giuliano Milanese is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 12 kilometres southeast of Milan. It received the honorary title of city with a presidential decree on April 24, 2000. The frazione of Viboldone is home to the historical Abbey of Viboldone. The town is served by the Borgolombardo and San Giuliano Milanese railway stations."
text9 = "Pioltello is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 7 kilometres northeast of Milan. Pioltello borders the following municipalities: Cernusco sul Naviglio, Vimodrone, Segrate, Rodano, Peschiera Borromeo, Vignate. Pioltello is served by Pioltello-Limito railway station. Among the churches, is the baroque-style, Roman Catholic Chiesa della Immacolata. Located about 6 kilometers east of Milan, the territory is between the SP ex SS 11 Padana Superiore and the SP 14 Rivoltana and is arranged along the north south axis, with an east west width of a couple of kilometers."
text10 = "Bollate is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 10 kilometres northwest of Milan. As of 30 November 2017, it had a population of 36,488. Bollate borders the following municipalities: Paderno Dugnano, Senago, Garbagnate Milanese, Arese, Cormano, Novate Milanese, Baranzate, Milan. Bollate received the honorary title of city, with a presidential decree on 11 October 1984. It is served by Bollate Centro railway station and Bollate Nord railway station. Sights include the historical Villa Arconati."
text11 = "Segrate is a town and comune located in the Metropolitan City of Milan in the Lombardy region of Northern Italy. An eastern suburb of Milan, in its area lies the airport of Milan Linate, the lake Idroscalo, the fair center of Novegro and the famous residential area Milano Due. Europark Idroscalo Milano is a popular amusement park that has been in existence since 1965. Although it is a small town, Segrate hosts large structures such as the CILEA Interuniversity Consortium, among the most advanced supercomputing centers in Europe, and the San Raffaele Hospital. Segrate received the honorary title of city with a presidential decree on 23 June 1989."
text12 = "Corsico is a comune in the Province of Milan in the Italian region Lombardy, bordering Milan on the southwest. Corsico received the honorary title of city with a presidential decree on 22 July 1987. Corsico is served by Corsico railway station."
text13 = "Cernusco sul Naviglio is a town and comune in the Metropolitan City of Milan, Lombardy, northwestern Italy. With a population of 33,436 as of 2015 it is the 14th-largest municipality in the metropolitan city. It is located 11 kilometres (6.8 mi) northeast of Milan along the Naviglio Martesana, which gives the town its name. The municipality of Cernusco sul Naviglio has a total area of 13.33 km2 with a median altitude of 133 metres above sea level."
text14 = "Abbiategrasso, formerly written Abbiate Grasso, is a comune and town in the Metropolitan City of Milan, Lombardy, northern Italy, situated in the Po valley approximately 22 kilometres from Milan and 38 kilometres from Pavia."
text15 = "San Donato Milanese is a comune in the Metropolitan City of Milan in the Italian region of Lombardy, located about 10 kilometres southeast of Milan. It is served by the San Donato underground station right on the borderline between the town and Milan and by the San Donato Milanese railway station, serving only trains for the Trenord S1 line 'Saronno–Lodi' and vice versa. Although the area was settled in ancient times, the origins of San Donato date back to the 7th century, when a pieve was founded here by the army of Grimoald I, Duke of Benevento. After a period under the Milanese family De Advocati, the town was a possession of the archbishops of Milan until the 16th century."
text16 = "Parabiago is a town located in the north-western part of the Metropolitan City of Milan, Lombardy, northern Italy. The town is crossed by the road to Sempione and Milan - Gallarate Railway; nearby flow the Olona river and the Canale Villoresi. Starting from the first Celtic-insubrian settlement, it developed during the Roman Empire rule, as documented by various archaeological discoveries of little objects, including the Parabiago Plate, a silver plate probably used to cover an ashes urn. "
text17 = "Buccinasco is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 7 kilometres southwest of Milan. The comune was created in 1841 through the merger of Buccinasco Castello, Rovido, Romano Banco and Gudo Gambaredo and, in 1871, Grancino and Ronchetto sul Naviglio. It remained an agricultural center until the 1950s. "
text18 = "Garbagnate Milanese is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 15 kilometre northwest of Milan. As of 30 November 2017, it had a population of 27.185. Garbagnate Milanese borders the following municipalities: Caronno Pertusella, Cesate, Lainate, Senago, Arese, Bollate, Baranzate. Garbagnate received the honorary title of city with a presidential decree on February 25, 1985."
text19 = "Bresso in the Metropolitan City of Milan in the Italian region Lombardy, located about 8 kilometres north of Milan. At the 2001 census the municipality had a population of 26,255 inhabitants and a population density of 8,027.2 persons/km², making it the most densely populated comune in Italy outside the Province of Naples. Bresso borders the following municipalities: Cinisello Balsamo, Cusano Milanino, Sesto San Giovanni, Cormano and Milan. Milan's general aviation airfield is located at Bresso and is the home of the Aero Club Milano and Aero Club Bresso."
text20 = "Lainate is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 15 kilometres northwest of Milan. Lainate borders the following municipalities: Caronno Pertusella, Origgio, Garbagnate Milanese, Nerviano, Arese, Rho, Pogliano Milanese. Lainate is home to Villa Visconti Borromeo Arese Litta, a Medici-inspired 1500s villa that today attracts many tourists because of its nymphaeum, and the headquarters of the confectionery company Perfetti Van Melle which sells candies and gums all over the world. It is also popular for Villoresi Canal and its forest, where people usually go for a walk or run."
text21 = "Magenta is a town and comune in the Metropolitan City of Milan in Lombardy, northern Italy. It became notable as the site of the Battle of Magenta in 1859. The color magenta takes its name from the battle. Magenta is the birthplace of Saint Gianna Beretta Molla and film producer Carlo Ponti The municipality of Magenta is part of the Parco naturale lombardo della Valle del Ticino, a Nature reserve included by UNESCO in the World Network of Biosphere Reserves."
text22 = "Cesano Boscone is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 6 kilometres southwest of Milan. Cesano Boscone borders the following municipalities: Milan, Corsico, Trezzano sul Naviglio. It is served by Cesano Boscone railway station."
text23 = "Peschiera Borromeo is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 12 kilometres southeast of Milan. It received the honorary title of city with a presidential decree on 6 August 1988. Peschiera Borromeo borders the following municipalities: Milan, Pioltello, Segrate, Rodano, Pantigliate, San Donato Milanese, Mediglia. The land was owned by the House of Borromeo of San Miniato in the 14th century and possibly earlier."
text24 = "Senago is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 13 kilometres north of Milan. As of 30 November 2017, it had a population of 21.519 and an area of 8.6 square kilometres. Senago borders the following municipalities: Limbiate, Cesate, Paderno Dugnano, Garbagnate Milanese, Bollate. The Villa San Carlo Borromeo is located in Senago. A historical residence, which was built in the XIV century, is immersed in a secular park of eleven hectares, 12 kilometres from Milan. It is the home town of Don Ambrogio Gianotti, a partigiano and the first priest of the church of St. Edward, Busto Arsizio"
text25 = "Trezzano sul Naviglio is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 9 km southwest of Milan. Trezzano sul Naviglio borders the municipalities of Buccinasco, Cusago, Cesano Boscone, Corsico, Gaggiano, Milan, and Zibido San Giacomo. It is served by Trezzano sul Naviglio railway station."
text26 = "Cornaredo is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 11 kilometres northwest of Milan. Cornaredo borders the following municipalities: Rho, Pregnana Milanese, Settimo Milanese, Bareggio, Cusago."
text27 = "Gorgonzola is a town in the Metropolitan City of Milan, Lombardy, northern Italy. It is part of the territory of the Martesana, north-east of Milan. Gorgonzola cheese is named after the town. The first written records mentioning the village of Gorgonzola date back to the 10th century: the notary clerk of the convent of Saint Ambrose in Milan was the caretaker of the church of Saints Gervasio and Protasio in 'Gorgontiola'."
text28 = "Cormano is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 9 kilometres north of Milan. Cormano borders the following municipalities: Paderno Dugnano, Bollate, Cusano Milanino, Bresso, Novate Milanese, Milan. It was previously served by the Cormano-Cusano Milanino railway station, which was closed in 2015, and replaced by the Cormano-Cusano Milanino railway station."
text29 = "Settimo Milanese is a comune in the Province of Milan in the Lombardy region of Italy. It is about 9 kilometres west of the city centre of Milan. The industrial district of Castelletto is home to Italtel and STMicroelectronics. Settimo Milanese borders Rho, Milan, Cornaredo, and Cusago. It's believed that the name comes from the distance between Settimo and Milan: it is in fact located near the seventh milestone of the road from Milan to Novara. The epithet 'Milanese' was added after the unification of Italy to distinguish it from other towns with the same name."
text30 = "Novate Milanese is a comune in the Metropolitan City of Milan in the Italian region Lombardy, located about 8 kilometres northwest of Milan. Novate Milanese borders the following municipalities: Bollate, Baranzate, Cormano, Milan. Novate received the honorary title of city with a presidential decree on 16 January 2004. Novate Milanese had a station on the Milano-Saronno railway and it is served by S1 and S3 lines of Milan Transportation System."
text31 = ""





#list of texts to select randomly
allTexts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12, text13, text14, text15, text16, text17, text18, text19, text20, text21, text22, text23, text24, text25, text26, text27, text28, text29, text30, text31, text32, text33, text34, text35, text36, text37, text38, text39, text40, text41, text42, text43, text44, text45]


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
  #Get maximum text width
  for ind in range(num_line-1):
    line_width, line_height = font.size(nlText[ind])
    if(line_width > max):
      max = line_width


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

      if (x >= 0 and x < sizeWidth - max):
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
  file1.close()'''
  time.sleep(0.3)
 


#Box text horizontal move
def verticalMove():

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

  text = rem_text(text, n, n*num_line)
  #text with new line
  nlText = text.split("#")
  #Maximum text width 
  max = 0
  #Get maximum text width and height
  for ind in range(num_line-1):
    line_width, line_height = font.size(nlText[ind])
    if(line_width > max):
      max = line_width

  char_width = max/n

  #Starting image position and speed
  x = (sizeWidth / 2) - (char_width*n/2)
  y = 0
  speed = 0.2
 
  '''# File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\Davide Mascheroni\\Desktop\\Risultati\\Results{}-Trial{}.txt".format(testernumber, index), "a")'''

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

      #5*(num_line) is a compensation for the space between each line
      if(y >= 0 and y < sizeHeight-(line_height*num_line) + 5*(num_line)):

        y = y + (1 * speed)
        screen.fill(BLACK)
        createVertBlock(x, y, font, nlText, dim_char)
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
  file1.write("\n\n VerticalMove END \n\n")
  file1.close()'''
  time.sleep(0.3)



#Box text that moves in diagonal
def diagMove():
 
  diag = math.sqrt((sizeWidth*sizeWidth)+(sizeHeight*sizeHeight))

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

  text = rem_text(text, n, n*num_line)
  #text with new line
  nlText = text.split("#")
  #Maximum text width 
  max = 0
  #Get maximum text width and height
  for ind in range(num_line-1):
    line_width, line_height = font.size(nlText[ind])
    if(line_width > max):
      max = line_width

  #Starting image position and speed
  x = 0
  y = 0
  speed = 0.2

  '''# File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\Davide Mascheroni\\Desktop\\Risultati\\Results{}-Trial{}.txt".format(testernumber, index), "a")'''

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

      if (x < sizeWidth - max):
        x = x + (1*(sizeWidth/diag) * speed)
        y = y + (1 *(sizeHeight/diag) * speed)
        screen.fill(BLACK)
        createVertBlock(x, y, font, nlText, dim_char)
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
  file1.write("\n\n DiagonalMove END \n\n")
  file1.close()'''
  time.sleep(0.3)



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
  file1.close()'''
  time.sleep(0.3)



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
  text = addSeparator(text, n, len(text))
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
  file1.close()'''
  time.sleep(0.3)


def main():
    
  pygame.init()
  pygame.mouse.set_visible(False)

  #shuffle the order of the animations
  tests_list = [horizontalMove, verticalMove, diagMove, horizontalScroll, verticalBlock]
  random.shuffle(tests_list)

  #run the animation after the shuffle
  for funct in tests_list:
    funct()
  
  pygame.quit()
  '''s.close()'''

if __name__ == "__main__":
    main()


'''LAB SPEEDS SETTINGS'''
#horizontalMove(), verticalMove() and diagMove() = 1
#horizontalScroll() = 1.4
#verticalBlock() = 0.5