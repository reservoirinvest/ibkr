from tkinter import Tk, Button
root = Tk()

def printName(event):
    print("Hello, my name is Kashi")

# button_1 = Button(root, text="Print my name", command=printName) # without binding

# Binding buttons
button_1 = Button(root, text="Print my name")
button_1.bind("<Button-1>", printName)
button_1.pack()

root.mainloop()