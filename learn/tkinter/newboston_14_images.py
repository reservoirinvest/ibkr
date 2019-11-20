# This is slightly different from newboston, who imports png file.
# We are import svg file here

import cairosvg as csvg

from tkinter import *

root = Tk()

img = csvg.svg2png(file_obj=open("activity.svg"))

photo = PhotoImage(data=img)
label = Label(root, image=photo)

label.pack()

root.mainloop()