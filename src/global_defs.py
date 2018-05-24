import numpy as np
from enum import IntEnum
from recordclass import recordclass

_point2d = recordclass('Point','x y')
obs = recordclass('Observation','allPos myInd preyInd')

class Point2D(_point2d):

    def __add__(self,other):
        x = self.x+other.x
        y = self.x+other.y
        return Point2D(x,y)

    def __sub__(self, other):
        "self-other"
        x = self.x - other.x
        y = self.y - other.y
        return Point2D(x,y)

    def __str__(self):
        return str(self.x)+str(self.y)

    def __hash__(self):
        return self.__str__()

    def __eq__(self, other):
        if ((self.x==other.x)and(self.y==other.y)):
            return True
        else:
            return False

    def manhattan_dist(self):
        return abs(self.x)+abs(self.y)



"""
static const Point2D VARIABLE_IS_NOT_USED MOVES[NUM_MOVES] = {Point2D(1,0),Point2D(-1,0),Point2D(0,1),Point2D(0,-1),Point2D(0,0)};
RIGHT,
    LEFT,
    UP,
    DOWN,
    NOOP,
"""
Actions = IntEnum('Actions','RIGHT LEFT UP DOWN NOOP')



