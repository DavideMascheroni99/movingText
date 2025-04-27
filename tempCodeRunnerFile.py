#Remove already used texts in the previous trials
    for i in range(1, trialN_int):
      mycursor.execute("SELECT txt FROM rem_index WHERE tester_number = %s and trial_number = %s", (tester_number, i))
      myresult = mycursor.fetchall()
      print(len (myresult))
      for j in range(len(myresult)):
        glb_var_const.allTexts.remove(myresult[j][0])