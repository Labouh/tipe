# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

NBINPUTS = 10 # 4 x 2 eyes + energy + speed
# SELF, PREY, PRED, PLANT (LEFT, RIGHT)
NBLOB = 6
NBNEUR = 4
NBOUTPUT = 2 # accelerate, turn

ENERGY = 500.
DIGEST = 10 #temps avant de pouvoir manger à nouveau

XMAP = YMAP = 500
SIZE = 5
VIEW = 100
WIDE = np.sin(1)		#Wide angle of view/vision périphérique
LOOKAT = np.sin(0.2)	#"Lookat" angle of view/vu par les 2 yeux


# RED eats GREEN eats BLUE that eats RED...
PLANT = 0

sigmoid = lambda x: np.exp(-np.logaddexp(0, -x)) # to avoid overflow / même si l'overflow était pas vraiment dérangeant 

# to compute difference on a round map
diffr = lambda x1, x2, xmax: min((x1-x2)%xmax, (x2-x1)%xmax)

class DNA:
    def __init__(self, dna = None):
        if dna:
            self.__dict__ = deepcopy(dna.__dict__)
            self.mutate()
        else:
            self.randomize()

    
    def randomize(self):
        self.iplace = np.random.randint(NBINPUTS//2, size=NBNEUR*NBLOB//2)#que1s inputs pris 
        self.iweight = 16. * np.random.ranf(size=NBNEUR*NBLOB//2) - 8. # poids des inputs
        self.lweight = 16. * np.random.ranf(size=(NBLOB, NBNEUR, NBNEUR)) - 8.
        self.oweight = 16. * np.random.ranf(size=(NBOUTPUT, NBNEUR*NBLOB)) - 8.
        self.color = np.random.randint(3) + 1 
        
    def mutate(self):
    #There is a disproportionately high chance of mutating the color
	#gene.  This reduces the chance of one color overwhelming the
	#others and causing the genepool to grow stagnant.
        rand = np.random.randint(100)
        if rand < 10:
            self.iplace[np.random.randint(len(self.iplace))] = np.random.randint(NBINPUTS//2)
        elif rand < 35:
            self.__mute(self.iweight)
        elif rand < 50:
            self.color = np.random.randint(3) + 1 
        elif rand < 75:
            self.__mute(self.lweight)
        else:
            self.__mute(self.oweight)            
            
    def __mute(self, arr):
        # mute a weight randomly
        arr.ravel()[np.random.randint(np.size(arr))] = 16. * np.random.ranf() - 8. #vue en ligne d'une matrice
        

class Input:
    def __init__(self, value = 0):
            self.axon = value
    def __mul__(self, other):
        'savoir multiplier par des poids (dentrites) utilisé par think'
        return self.axon * other

class Neuron(Input): 
    def __init__(self, parents, dentrites):
        'un neuronne contient ses parents et le poids des dentrites'
        self.parents = parents
        self.dentrites = dentrites
    def think(self):
        'reflexion: somme des multiplications des parents par les dentrites'
        self.axon = sigmoid(np.dot(self.parents, self.dentrites))
    
class Lobe:
    # un lobe = 2 couches de 4 neurones
    def __init__(self, inputs, iplace, iweight, lweight):
        #partie des tableaux du dna qui concernent ce lobe
            self.top = []
            self.bottom = []
            # creation des neurones d'entrée du lobe en lien avec les inputs
            for (i, place) in enumerate(iplace): #retourne la valeur et la position
                self.top.append(Neuron([inputs[place*2]], [iweight[i]]))
                self.top.append(Neuron([inputs[place*2 + 1]], [iweight[i]]))
            # création du niveau bas en lien avec les neurones du haut
            for bot in range(NBNEUR):
                self.bottom.append(Neuron(self.top, lweight[bot]))
        
    def think(self):
        ' reflexion du lob : le haut puis le bas'
        for n in self.top:
            n.think()
        for n in self.bottom:
            n.think()
        
class Brain:
    def __init__(self, inputs, dna):
        self.lobes = []
        self.output = []
        neurons = [] # listes des neurones en liens avec les outputs
        
        # creation des lobes
        for l in range(NBLOB):
            self.lobes.append(Lobe(inputs,
                                   dna.iplace[l*NBNEUR//2:(l+1)*NBNEUR//2],
                                   dna.iweight[l*NBNEUR//2:(l+1)*NBNEUR//2],
                                   dna.lweight[l]))
            neurons.extend(self.lobes[-1].bottom) #concatène sur place
        # creation des neurones de sortie
        for o in range(NBOUTPUT):
            self.output.append(Neuron(neurons, dna.oweight[o])) 
            
    def think(self):
        for l in self.lobes:
            l.think()
        for n in self.output:
            n.think()

class Element:
    'class vide juste pour définir kill qui ne fait rien'
    def kill(self):
        """do what is needed when the element disapear"""
        pass # by default nothing
            
class Plant(Element):
    def __init__(self):
        self.x = np.random.randint(XMAP)
        self.y = np.random.randint(YMAP)
        self.color = PLANT
        self.energy = 1
        
    def collide(self, other):
        if other.color != PLANT:
            other.collide(self)
            
    
class Animal(Element):
    def __init__(self, world, father = None):
        self.world = world
        self.inputs = [Input() for i in range(NBINPUTS)]
        self.speed = 0
        if father:        
            self.x = father.x
            self.y = father.y
            self.route = father.route + np.pi
            self.dna = DNA(father.dna)
        else:
            self.x = np.random.randint(XMAP)
            self.y = np.random.randint(YMAP)
            self.route = 2. * np.pi * np.random.ranf()
            self.dna = DNA()
        self.brain = Brain(self.inputs, self.dna)
        self.energy = ENERGY
        self.digest = DIGEST
        self.color = self.dna.color
    
    def move(self):
        self.see()
        self.brain.think()
        self.speed = self.brain.output[0].axon * SIZE
        self.route += self.brain.output[1].axon - 0.5
        self.energy -= abs(self.speed) / 2. + 1
        self.x += self.speed * np.cos(self.route)
        self.y += self.speed * np.sin(self.route)
        self.x = self.x % XMAP
        self.y = self.y % YMAP
        self.digest = max(0, self.digest - 1)
        
        # clone ?
        if self.energy > 2*ENERGY:
            self.energy -= ENERGY
            # create a new animal on the same class (allo inheritance)
            self.world.append(self.__class__(self.world, self))
    
    def see(self):
        # clear
        for i in range(len(self.inputs)):
            self.inputs[i].axon = 0.
        for specie in self.world:
            if specie is not self:
                dx = diffr(specie.x, self.x, XMAP)
                dy = diffr(specie.y, self.y, YMAP)
                r2 = dx * dx + dy * dy
                if r2 < VIEW*VIEW:
                    da = np.arctan2(dy, dx) - self.route 
                    if np.cos(da) > 0: #behind you?
                        # decide the input depending on the species
                        # 0,1: same, 2,3: prey, 4,5: pred and 6, 7: plant
                        if specie.color == PLANT:
                            i = 6
                        else:
                            i = 2 * ((specie.color - self.color)%3)
                        f = 1.0 - (r2 - SIZE*SIZE) / (VIEW*VIEW - SIZE*SIZE)
                        sa = np.sin(da)
                        # check view angle
                        if sa < WIDE and sa > -LOOKAT: #left
                            self.inputs[i].axon += f
                        if sa < LOOKAT and sa > -WIDE:#right
                            self.inputs[i+1].axon += f
                            
        self.inputs[8].axon = self.energy / ENERGY
        self.inputs[9].axon = self.speed / SIZE
        
    def collide(self,other):
        energy = 0
        if other.color == PLANT:
            energy = 500
        else:
            dif = (other.color - self.color)%3
            if dif == 1 and not self.digest:
                self.digest = DIGEST
                energy = 400
            elif dif == 2 and not other.digest:
                other.digest = DIGEST
                energy = -400
        self.energy += energy
        other.energy -= energy
                

class World:
    def __init__(self, A = Animal, P = Plant): # inject dependency for inheritance                
        self.curve = []
        self.world= []
        # add new animals in the world
        for i in range(100):        
            self.world.append(A(self.world))
        for i in range(50):        
            self.world.append(P())
            self.P = P # to add plant during run

    def run(self, time):
        for t in range(time): # time
            if np.random.ranf() > 0.8:
                self.world.append(self.P())
            for w in [x for x in self.world if x.color != PLANT]:
                w.move()
            for i in range(len(self.world)):
                for j in range(i, len(self.world)):
                    dx = diffr(self.world[i].x, self.world[j].x, XMAP)
                    dy = diffr(self.world[i].y, self.world[j].y, YMAP)
                    if (dx * dx + dy * dy) < SIZE*SIZE:
                        self.world[i].collide(self.world[j])
        
            i = 0
            while i < len(self.world):
                if self.world[i].energy < 0:
                    self.world[i].kill()
                    del self.world[i]
                else:
                    i+=1
            self.curve.append(np.bincount([x.color for x in self.world], minlength=4))
            

    