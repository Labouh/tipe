# -*- coding: utf-8 -*-
"""
Display of neural animals with tkinter (Darwin)
use v, f to swithc view
    p, s, g to pause, go step by step and restart
    q to quit
    mousewheel to zoom
neural.sav saves the content of the world (remove the file to restart from a new world)
"""

import neural
import numpy as np
import pickle

COLOR=['gray','red','green','blue']
def create_poly(coord, **kwargs):
    return canvas.create_polygon([mscale * c for c in coord], **kwargs)
def create_oval(*coord, **kwargs):
    return canvas.create_oval(*[mscale * c for c in coord], **kwargs)
def create_line(*coord, **kwargs):
    return canvas.create_line(*[mscale * c for c in coord], **kwargs)


class Animal(neural.Animal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if fullview:
            self.draw()

    def kill(self):
        if fullview:
            canvas.delete(self.poly)
    
    def move(self):
        super().move()
        if fullview:
            canvas.delete(self.poly)
            self.draw()
    
    def draw(self):
        self.poly = create_poly([self.x, self.y, 
                                self.x - neural.SIZE * np.cos(self.route) + neural.SIZE * np.sin(self.route)/2,
                                self.y - neural.SIZE * np.sin(self.route) - neural.SIZE * np.cos(self.route)/2,
                                self.x - neural.SIZE * np.cos(self.route) - neural.SIZE * np.sin(self.route)/2,
                                self.y - neural.SIZE * np.sin(self.route) + neural.SIZE * np.cos(self.route)/2],
            outline=COLOR[self.color], 
            fill=COLOR[self.color])
                                          

class Plant(neural.Plant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if fullview:
            self.draw()
    def kill(self):
        super().kill()
        if fullview:
            canvas.delete(self.circle)
    def draw(self):
        if fullview:
            self.circle = create_oval(self.x - neural.SIZE//2, self.y - neural.SIZE//2,
                                      self.x + neural.SIZE//2, self.y + neural.SIZE//2,
            outline='gray', 
            fill='gray')

class World(neural.World):
    def __init__(self, A, P):
        # retreive from the saved file
        try:
            self.pplant = 0.2
            with open('neural.sav', 'rb') as savfile:
               self.world = pickle.load(savfile)
            self.P = P
            self.full()
        except IOError:
            # or start a new world
            super().__init__(A, P)
        self.go = True

    def display(self):
        super().run(1)

        if not fullview:
            self.localview()
        else:
            canvas.delete('legend')
            canvas.create_text(mscale * (neural.XMAP/2 + 80), mscale *(neural.YMAP - 15), 
                text='plant:%02d, red: %02d, green:%02d, blue:%02d, %%plant:%.1f' 
                    % (tuple(np.bincount([x.color for x in self.world], minlength=4))+(self.pplant * 100,)),
                    tags='legend')
        if self.go: 
            canvas.after(10,self.display)

    def pause(self):
        "pause"
        self.go = False
    
    def start(self):
        "start when in pause"
        if not self.go:
            self.go = True
            self.display()

    def step(self):
        "one step forward when in pause"
        if not self.go:
            self.display()

    def local(self):
        "switch to local view"
        global fullview
        fullview = False
        self.localview()
    
    def localview(self):
        "draw a relative view from on animal"
        # first clear screen
        canvas.delete("all")
        # find the first animal that we will follow
        for w in self.world:
            if w.color != neural.PLANT:
                current = w
                break

        # scale regarding the full view
        scale = neural.XMAP / neural.VIEW / 2
        # animal that we follow in the bottom of the screen
        x0 = neural.XMAP/2
        y0 = neural.YMAP - 5 * neural.SIZE * scale

        # animals (including current!) and plants on relative view
        for w in self.world:
            dx = neural.diffr(current.x, w.x, neural.XMAP)
            dy = neural.diffr(current.y, w.y, neural.YMAP)
            if abs(dx) < neural.VIEW and abs(dy) < neural.VIEW:
                # rotate to have a relative view
                x = x0 + scale * (dx * np.sin(current.route) - dy * np.cos(current.route))
                y = y0 + scale * (dy * np.sin(current.route) + dx * np.cos(current.route))
                if w.color == neural.PLANT:
                    create_oval(x - scale * neural.SIZE//2, y - scale * neural.SIZE//2,
                                x + scale * neural.SIZE//2, y + scale * neural.SIZE//2,
                    outline='gray', 
                    fill='gray') 
                else:
                    dr = w.route - current.route - np.pi / 2
                    create_poly([x, y, 
                        x - scale * (neural.SIZE * np.cos(dr) + neural.SIZE * np.sin(dr)/2),
                        y - scale * (neural.SIZE * np.sin(dr) - neural.SIZE * np.cos(dr)/2),
                        x - scale * (neural.SIZE * np.cos(dr) - neural.SIZE * np.sin(dr)/2),
                        y - scale * (neural.SIZE * np.sin(dr) + neural.SIZE * np.cos(dr)/2)],
                    outline=COLOR[w.color], 
                    fill=COLOR[w.color])
        create_line(x0, y0, x0 - scale * neural.VIEW * neural.WIDE, y0 - scale * neural.VIEW*(1 - neural.WIDE**2), dash=(2,4), fill='orange')
        create_line(x0, y0, x0 + scale * neural.VIEW * neural.WIDE, y0 - scale * neural.VIEW*(1 - neural.WIDE**2), dash=(2,4), fill='cyan')
        create_line(x0, y0, x0 + scale * neural.VIEW * neural.LOOKAT, y0 - scale * neural.VIEW*(1 - neural.LOOKAT**2), dash=(2,4), fill='orange')
        create_line(x0, y0, x0 - scale * neural.VIEW * neural.LOOKAT, y0 - scale * neural.VIEW*(1 - neural.LOOKAT**2), dash=(2,4), fill='cyan')
        # draw an energy bar
        energybar = [10, neural.YMAP - 5, 150, neural.YMAP - 5, 150, neural.YMAP - 20, 10, neural.YMAP - 20 ]
        create_poly(energybar, outline = 'gray')
        energybar[2] = energybar[4] = energybar[0] + current.energy / neural.ENERGY * (energybar[2] - energybar[0]) / 2
        create_poly(energybar, outline = 'gray', fill = 'gray')        
        # and information regarding current animal
        canvas.create_text(mscale * (neural.XMAP/2 + 80), mscale *(neural.YMAP - 15), 
            text='age: %04d, children: %02d, plant:%02d, red: %02d, green:%02d, blue:%02d ' 
                % ((current.age, current.children)+tuple(np.bincount([x.color for x in self.world], minlength=4))))

        # draw the brain 
        # ensure inputs are on current place
        current.see()
        ix = 150
        iy = ns = 10 # size of a neuron on screen
        s = current.color - 1
        # first inputs, ie eyes (with the good color)
        for (i, p) in enumerate(current.inputs):
            if i < 6:
                c = COLOR[s + 1]
                s = (s + i%2)%3 
            elif i < 8:
                c = COLOR[0]
            else:
                c = 'black'
            create_oval(ix, iy, ix+ns, iy+ns, fill="#%02x%02x%02x" % (3*(max(0,int(255*(1-p.axon/2))),)), outline=c)
            ix += 2*ns
        ix0 = 50
        iy0 = iy + 3*ns
        colsyn = lambda w: "#%02xaaaa" % (128-int(128*w/8)) if w <0 else "#aa%02xaa" % (128+int(128*w/8))
        for (i, l) in enumerate(current.brain.lobes):
            ix = ix0
            iy = iy0
            for (ni, n) in enumerate(l.top):
                # top neurons
                create_oval(ix, iy, ix+ns, iy+ns, fill="#%02x%02x%02x" % (3*(int(255*(1-n.axon)),)))
                # and synapse to inputs
                li = current.dna.inputs[i*neural.NBNEUR//2+ni//2]*2
                wi = current.dna.iweight[i*neural.NBNEUR//2][ni//2]
                if wi != 0:
                    create_line(150 + ns * (li + ni%2) * 2 + ns/2, 20, ix + ns/2 , iy, fill=colsyn(wi))
                ix += 1.5*ns
            iy += 2*ns
            ix = ix0
            for (ni, n) in enumerate(l.bottom):
                # bottoms neurons
                create_oval(ix, iy, ix+ns, iy+ns, fill="#%02x%02x%02x" % (3*(int(255*(1-n.axon)),)))
                # and synapse to top neurons
                for (nt, wi) in enumerate(current.dna.lweight[i][ni][:-1]): # weight to top neuron removing bias
                    if wi != 0:
                        create_line(ix0 + ns * nt * 1.5 + ns/2, iy - ns, ix + ns/2 , iy, fill=colsyn(wi))
                ix += 1.5*ns
            ix0 = ix + ns
        ix = neural.XMAP / 2 - 2*ns
        iy = iy + 3*ns
        for (no, n) in enumerate(current.brain.output):
            create_oval(ix, iy, ix+ns, iy+ns, fill="#%02x%02x%02x" % (3*(int(255*(1-n.axon)),)))
            for (nt, wo) in enumerate(current.dna.oweight[no]):
                if wo != 0:
                    create_line(50 + ns * nt * 1.5 + ns/2 + ns * nt//4, iy - 2*ns, ix + ns/2 , iy, fill=colsyn(wo))
            ix += 2*ns

    def full(self):
        "switch to full world view"
        canvas.delete("all")
        global fullview
        fullview = True
        for w in self.world:
            w.draw()

    def quit(self):
        "save the world to neural.sav and quit"
        self.go = False
        with open('neural.sav', 'wb') as savfile:
            pickle.dump(self.world ,savfile, pickle.HIGHEST_PROTOCOL)
        root.destroy()

    def plus(self):
        "more plants"
        self.pplant += (1 - self.pplant)*0.1
    def moins(self):
        "less plants"
        self.pplant -= self.pplant * 0.1

    def keyup(self, e):
        "catch key to change display"
        keys = {
            'q': self.quit,
            'v': self.local,
            'f': self.full,
            'p': self.pause,
            's': self.step,
            'g': self.start,
            'plus': self.plus,
            'minus': self.moins,
            'Return': lambda:None,
        }
        def help():
            msg = "Use :\n  mouse wheel to zoom/unzoom\n"
            for (k, f) in keys.items():
                msg += '  %s: %s\n' %(k, f.__doc__)
            messagebox.showinfo("Help", msg)

        keys.get(e.keysym, help)()


from tkinter import Tk, Canvas, messagebox

mscale = 1
def zoomer(event):
    global mscale
    if (event.delta > 0):
        mscale *= 1.1
        canvas.config(width=neural.XMAP * mscale, height=neural.YMAP * mscale)
        canvas.scale("all", 0, 0, 1.1, 1.1)
    elif (event.delta < 0):
        mscale *= 0.9
        canvas.config(width=neural.XMAP * mscale, height=neural.YMAP * mscale)
        canvas.scale("all", 0, 0, 0.9, 0.9)

root = Tk()
root.title("Darwin")
root.resizable(0, 0)

canvas = Canvas(root, width=neural.XMAP, height=neural.YMAP, bd=0, highlightthickness=0, background='white')
canvas.pack()
# full view to see all creatures, else we follows one (global ugly variable)
fullview = True
world = World(Animal, Plant)
canvas.bind("<KeyRelease>", world.keyup)
canvas.focus_set()
canvas.bind("<MouseWheel>",zoomer)
# Hack to make zoom work on Windows
root.bind_all("<MouseWheel>",zoomer)
canvas.after(100,world.display)
root.mainloop()