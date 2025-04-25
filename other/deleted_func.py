'''#Box text vertical move
def horizontalMove(txt):
  
  #dimension of a single character
  dim_char = 30
  #number of a box lines
  num_line = 8
  #Number of characters per line
  n = 40

  #Create a font
  font = pygame.font.SysFont("Arial", dim_char)
  #Text to show
  text = txt
  # Setting the time
  test_time = 10
  t_end = time.time() + test_time

  #Starting image position and speed
  x = 0
  y = (sizeHeight / 2) - ((num_line*dim_char)/2)
  speed = 0.2

  text = rem_text(text, n, n*num_line, num_line)
  #text with new line
  nlText = text.split("#")
  max = 0
  #Get maximum text width
  for ind in range(num_line-1):
    line_width, line_height = font.size(nlText[ind])
    if(line_width > max):
      max = line_width


  
  # File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\Davide Mascheroni\\Desktop\\Risultati\\Results{}-Trial{}.txt".format(tester_number, trial_number), "a")
  

  wall = False

  while time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()

    while not wall and time.time() <= t_end:

      # Sending data to the server and writing it on the respective file
      casual_data = s.recv(1024)
      file1.write(bytes.decode(casual_data))

      if (x >= 0 and x < sizeWidth - max):
        x = x + (1 * speed)
        screen.fill(BLACK)
        createVertBlock(x, y, font, nlText, dim_char)
        pygame.display.flip()
      else:
        wall = True

    while wall and time.time() <= t_end:

      # Sending data to the server and writing it on the respective file
      casual_data = s.recv(1024)
      file1.write(bytes.decode(casual_data))

      if(x > 1*speed):
        x = x - (1 * speed)
        screen.fill(BLACK)
        createVertBlock(x, y, font, nlText, dim_char)
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
  file1.write("\n\n HorizontalMove end\n\n")
  file1.close()
  time.sleep(0.3)
 


#Box text horizontal move
def verticalMove(txt):

  #dimension of a single character
  dim_char = 30
  #number of a box lines
  num_line = 8
  #Number of characters per line
  n = 40

  #Create a font
  font = pygame.font.SysFont("Arial", dim_char)
  #Text to show
  text = txt
  # Setting the time
  test_time = 10
  t_end = time.time() + test_time

  text = rem_text(text, n, n*num_line, num_line)
  #text with new line
  nlText = text.split("#")
  #Maximum text width 
  max = 0
  
  line_width, line_height = font.size(nlText[0])


  #Starting image position and speed
  x = (sizeWidth / 2) - (line_width/2)
  y = 0
  speed = 0.2
 
  # File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\Davide Mascheroni\\Desktop\\Risultati\\Results{}-Trial{}.txt".format(tester_number, trial_number), "a")

  wall = False

  while time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()
  
    while not wall and time.time() <= t_end:

      # Sending data to the server and writing it on the respective file
      casual_data = s.recv(1024)
      file1.write(bytes.decode(casual_data))

      #5*(num_line) is a compensation for the space between each line
      if(y >= 0 and y < sizeHeight-(line_height*num_line) + 5*(num_line)):

        y = y + (1 * speed)
        screen.fill(BLACK)
        createVertBlock(x, y, font, nlText, dim_char)
        pygame.display.flip()
      else:
        wall = True

    while wall and time.time() <= t_end:

      # Sending data to the server and writing it on the respective file
      casual_data = s.recv(1024)
      file1.write(bytes.decode(casual_data))

      if(y > 1*speed):
        y = y - (1 * speed)
        screen.fill(BLACK)
        createVertBlock(x, y, font, nlText, dim_char)
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
  file1.write("\n\n VerticalMove END \n\n")
  file1.close()
  time.sleep(0.3)



#Box text that moves in diagonal
def diagMove(txt):
 
  diag = math.sqrt((sizeWidth*sizeWidth)+(sizeHeight*sizeHeight))

  #dimension of a single character
  dim_char = 30
  #number of a box lines
  num_line = 8
  #Number of characters per line
  n = 40

  #Create a font
  font = pygame.font.SysFont("Arial", dim_char)
  #Text to show
  text = txt
  # Setting the time
  test_time = 10
  t_end = time.time() + test_time

  text = rem_text(text, n, n*num_line, num_line)
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

  # File to write on
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))
  file1 = open("C:\\Users\\Davide Mascheroni\\Desktop\\Risultati\\Results{}-Trial{}.txt".format(tester_number, trial_number), "a")

  wall = False

  while time.time() <= t_end:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        sys.exit()


    while not wall and time.time() <= t_end:

      # Sending data to the server and writing it on the respective file
      casual_data = s.recv(1024)
      file1.write(bytes.decode(casual_data))

      if (x < sizeWidth - max):
        x = x + (1*(sizeWidth/diag) * speed)
        y = y + (1 *(sizeHeight/diag) * speed)
        screen.fill(BLACK)
        createVertBlock(x, y, font, nlText, dim_char)
        pygame.display.flip()
      else:
        wall = True

    while wall and time.time() <= t_end:

      # Sending data to the server and writing it on the respective file
      casual_data = s.recv(1024)
      file1.write(bytes.decode(casual_data))
      
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

 # Sending data to the server and writing it on the respective file
  s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="0" />\r\n'))
  time.sleep(0.3)
  casual_data = s.recv(1024)
  time.sleep(0.3)
  file1.write(bytes.decode(casual_data))
  file1.write("\n\n DiagonalMove END \n\n")
  file1.close()
  time.sleep(0.3)
  
  
  
  #Finish the block text after the last point
def rem_text(text, n, max_lenght, num_line):
  txt = addSeparator(text, n, max_lenght)
  s = list(txt)
  point = find_last_point(txt)
  if (point != -1 and point > ((num_line-1)*n)):
    point = point + 1
    for i in range(len(txt)):
      if (i == point):
        fin = ''.join(s[0:i])
        return fin
  return txt
  

  
#find the last point of a block
def find_last_point(text):
  point = text.rfind('.')
  return point

  
  '''