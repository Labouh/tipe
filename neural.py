# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

NBINPUTS = 10 # 4 x 2 eyes + energy + speed
# SELF, PREY, PRED, PLANT (LEFT, RIGHT)
NBLOB = 6
NBNEUR = 4
NBOUTPUT = 2 # accelerate, turn

ENERGY = 500.
DIGEST = 3
MATURITY = 1000

XMAP = YMAP = 500
SIZE = 5
VIEW = 100
WIDE = np.sin(1)		#Wide angle of view
LOOKAT = np.sin(0.2)	#"Lookat" angle of view


# RED eats GREEN eats BLUE that eats RED...
PLANT = 0

sigmoid = lambda x: np.exp(-np.logaddexp(0, -x)) # to avoid overflow

# to compute difference on a round map
diffr = lambda x1, x2, xmax: (x1-x2)-np.sign(x1-x2)*xmax if abs(x1-x2) > xmax/2  else (x1-x2)

class DNA:
    def __init__(self, dna = None, dna2 = None):
        if dna:
            # copy the first parent
            self.__dict__ = deepcopy(dna.__dict__)
            if dna2:
                # sexual reproduction with 2 parents
                self.merge(self.inputs, dna2.inputs)
                self.merge(self.iweight, dna2.iweight)
                self.merge(self.lweight, dna2.lweight)
                self.merge(self.oweight, dna2.oweight)
            else:
                # clonage with mutation
                self.mutate()
        else:
            self.randomize()

    def merge(self, arr, arr2):
        for i in range(arr.size):
            if np.random.randint(100) >= 50:
                arr.ravel()[i] = arr2.ravel()[i]
    
    def randomize(self):
        self.inputs = np.random.randint(NBINPUTS//2, size=NBNEUR*NBLOB//2)
        self.iweight = 16. * np.random.ranf(size=(NBNEUR*NBLOB//2, 2)) - 8.
        self.lweight = 16. * np.random.ranf(size=(NBLOB, NBNEUR, NBNEUR+1)) - 8.
        self.oweight = 16. * np.random.ranf(size=(NBOUTPUT, NBNEUR*NBLOB+1)) - 8.
        self.color = np.random.randint(3) + 1 
        
    def mutate(self):
    #There is a disproportionately high chance of mutating the color
	#gene.  This reduces the chance of one color overwhelming the
	#others and causing the genepool to grow stagnant.
        rand = np.random.randint(100)
        if rand < 10:
            self.inputs[np.random.randint(len(self.inputs))] = np.random.randint(NBINPUTS//2)
        elif rand < 35:
            self.__mute(self.iweight)
        elif rand < 50:
            self.color = (self.color + np.random.randint(2)) % 3 + 1 
        elif rand < 75:
            self.__mute(self.lweight)
        else:
            self.__mute(self.oweight)            
            
    def __mute(self, arr):
        # mute a weight randomly
        if np.random.ranf() < 0.5: #50% chance of pruning a synapse
            arr.ravel()[np.random.randint(np.size(arr))] = 0
        else:
            arr.ravel()[np.random.randint(np.size(arr))] = 16. * np.random.ranf() - 8.

        

class Input:
    def __init__(self, value = 0):
            self.axon = value
    def __mul__(self, other):
        return self.axon * other

class Neuron(Input):
    def __init__(self, parents, dentrites):
        self.parents = parents
        self.dentrites = dentrites
    def think(self):
        self.axon = sigmoid(np.dot(self.parents + [1], self.dentrites))
    
class Lobe:
    # a lobe of 4 neurons with 2 layers
    def __init__(self, inputs, iplace, iweight, lweight):
            self.top = []
            self.bottom = []
            for (i, place) in enumerate(iplace):
                self.top.append(Neuron([inputs[place*2]], iweight[i]))
                self.top.append(Neuron([inputs[place*2 + 1]], iweight[i]))
            for bot in range(NBNEUR):
                self.bottom.append(Neuron(self.top, lweight[bot]))
        
    def think(self):
        for n in self.top:
            n.think()
        for n in self.bottom:
            n.think()
        
class Brain:
    def __init__(self, inputs, dna):
        self.lobes = []
        self.output = []
        neurons = []
        
        for l in range(NBLOB):
            self.lobes.append(Lobe(inputs,
                                   dna.inputs[l*NBNEUR//2:(l+1)*NBNEUR//2],
                                   dna.iweight[l*NBNEUR//2:(l+1)*NBNEUR//2],
                                   dna.lweight[l]))
            neurons.extend(self.lobes[-1].bottom)
        for o in range(NBOUTPUT):
            self.output.append(Neuron(neurons, dna.oweight[o])) 
            
    def think(self):
        for l in self.lobes:
            l.think()
        for n in self.output:
            n.think()

class Element:
    def kill(self):
        "do wat is needed when the element disapear"
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
    """
    >>> world = []
    >>> a1 = Animal(world)
    >>> a1.x = a1.y = 0
    >>> a1.route = np.pi/2
    >>> a2 = Animal(world)
    >>> a2.x = XMAP - 10
    >>> a2.y = 10
    >>> a3 = Animal(world)
    >>> a3.x = 10
    >>> a3.y = 10
    >>> a1.color =  a2.color = a3.color = 1
    >>> p = Plant()
    >>> p.x = 0
    >>> p.y = 10
    >>> world.extend([a1,a2, a3, p])
    >>> a1.see()
    >>> ['{:.2}'.format(x.axon) for x in a1.inputs]
    ['0.98', '0.98', '0.0', '0.0', '0.0', '0.0', '0.99', '0.99', '1.0', '0.0']
    >>> a1.move()
    >>> [x.axon > 0 and x.axon < 1 for x in a1.brain.output]
    [True, True]
    """

    def __init__(self, world, father = None, mother = None):
        self.world = world
        self.inputs = [Input() for i in range(NBINPUTS)]
        self.speed = 0
        self.pregnant = MATURITY
        if father:
            self.name = father.name + ',' + str(father.children)
            self.x = father.x
            self.y = father.y
            self.route = father.route + np.pi
            if mother:
                self.dna = DNA(father.dna, mother.dna)
            else:
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
        self.children = 0
        self.age = 0

    
    def move(self):
        self.see()
        self.brain.think()
        self.speed = self.brain.output[0].axon * SIZE
        self.route = (self.route + self.brain.output[1].axon - 0.5) % (2 * np.pi)
        self.energy -= abs(self.speed) / 2. + 1
        self.x += self.speed * np.cos(self.route)
        self.y += self.speed * np.sin(self.route)
        self.x = self.x % XMAP
        self.y = self.y % YMAP
        self.digest = max(0, self.digest - 1)
        self.pregnant = max(0, self.pregnant - 1)
        
        # clone ?
        if self.energy > 2*ENERGY:
            self.energy = ENERGY
            # create a new animal on the same class (allo inheritance)
            self.world.append(self.__class__(self.world, self))
            self.children += 1
        self.age += 1
    
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
                        f = 1.0 - r2 / (VIEW*VIEW)
                        sa = np.sin(da)
                        # check view angle
                        if sa < WIDE and sa > -LOOKAT: #right
                            self.inputs[i+1].axon += f
                        if sa < LOOKAT and sa > -WIDE: #left
                            self.inputs[i].axon += f
                            
        self.inputs[8].axon = self.energy / ENERGY
        self.inputs[9].axon = self.speed / SIZE
        
    def collide(self,other):
        energy = 0
        if other.color == PLANT:
            energy = 500
        elif other.color == self.color and self is not other: # sexual reproduction
            if self.pregnant == 0 and other.pregnant == 0 \
                and self.energy > ENERGY/2 and other.energy > ENERGY/2: # only if not done recently
                self.pregnant = MATURITY
                other.pregnant = MATURITY
                self.children += 1
                other.children += 1
                self.world.append(self.__class__(self.world, self, other))
                # need some energy
                self.energy -= 10
                other.energy -= 10
        else: # eat !
            dif = (other.color - self.color)%3
            if dif == 1 and not self.digest:
                self.digest = DIGEST
                energy = 500
            elif dif == 2 and not other.digest:
                other.digest = DIGEST
                energy = -500
        self.energy += energy
        other.energy -= energy
                

class World:
    def __init__(self, A = Animal, P = Plant): # inject dependency for inheritance                
        self.world=[]
        # proba of growing a new plant each turn
        self.pplant = 0.2
        # add new animals in the world
        for i in range(100):   
            a = A(self.world)
            a.name = str(i)     
            self.world.append(a)
        for i in range(50):        
            self.world.append(P())
            self.P = P # to add plant during run

    def run(self, time):
        for t in range(time): # time
            if np.random.ranf() > 1 - self.pplant:
                self.world.append(self.P())
            animals = [x for x in self.world if x.color != PLANT]
            for w in animals:
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
            
if __name__ == "__main__":
    import doctest
    doctest.testmod()

    