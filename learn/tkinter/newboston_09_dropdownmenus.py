from tkinter import *

def doNothing():
    print("ok ok I won't...")

root = Tk()

menu = Menu(root)
root.config(menu=menu)

subMenu = Menu(menu, tearoff=0)
menu.add_cascade(label="File", menu=subMenu)
subMenu.add_command(label="New project...", command=doNothing)
subMenu.add_command(label="New", command=doNothing)
subMenu.add_separator()
# subMenu.add_command(label="Exit", command=doNothing)
subMenu.add_command(label="Exit", command=root.quit)

editMenu = Menu(menu, tearoff=0)
menu.add_cascade(label="Edit", menu=editMenu)
editMenu.add_command(label="Redo", command=doNothing)

root.mainloop()