# -*- coding: utf-8 -*-
"""
"""

import neural
import numpy as np


class Animal(neural.Animal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draw()

    def kill(self):
        canvas.delete(self.poly)
    
    def move(self):
        super().move()
        canvas.delete(self.poly)
        self.draw()
    
    def draw(self):
        color =  "#%02x%02x%02x" % (int(255*self.diet),10,10)
        self.poly = canvas.create_polygon([self.x, self.y, 
                                           self.x - neural.SIZE * np.cos(self.route) + neural.SIZE * np.sin(self.route)/2,
                                           self.y - neural.SIZE * np.sin(self.route) - neural.SIZE * np.cos(self.route)/2,
                                           self.x - neural.SIZE * np.cos(self.route) - neural.SIZE * np.sin(self.route)/2,
                                           self.y - neural.SIZE * np.sin(self.route) + neural.SIZE * np.cos(self.route)/2],
            outline=color, 
            fill=color)
                                          

class Plant(neural.Plant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draw()
    def kill(self):
        canvas.delete(self.circle)
    def draw(self):
        self.circle = canvas.create_oval(self.x - neural.SIZE//2, self.y - neural.SIZE//2,
                                         self.x + neural.SIZE//2, self.y + neural.SIZE//2,
            outline='green', 
            fill='green')

class World(neural.World):
    def display(self):
        super().run(1)
        canvas.after(10,self.display)
        

from tkinter import Tk, Canvas


root = Tk()
root.title("Darwin")
root.resizable(0, 0)
#root.wm_attributes("-topmost", 1)

canvas = Canvas(root, width=neural.XMAP, height=neural.YMAP, bd=0, highlightthickness=0)
canvas.pack()

world = World(Animal, Plant)
canvas.after(100,world.display)
root.mainloop()