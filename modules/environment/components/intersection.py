from math import pi
from .visual import Lane

class TrafficLane():
    """
    An abstract data type used to represent a traffic lane
    All coordinates are global. (0,0) is the top left corner.
    """

    corners = []

    def __init__(self,corners,shapes):
        """Create a new lane which with polygon corners"""
        self.corners = corners
        self.shapes = shapes
        self.shapes.insert(0,Lane(corners))



class Intersection():
    """
    An abstract data type used to represent an intersection
    All coordinates are global. (0,0) is the top left corner.

    lanes: Lane boundaries - used for calculations
    shapes: Lane boundaries - used for rendering only
    """

    lanes = []
    shapes = []
    width = 800
    height = 800

    def __init__(self,lanes):
        """Create a new intersection model with a set of lanes"""
        self.lanes = lanes

        # Lanes are shapes too
        for lane in lanes:
            for shape in lane.shapes:
                self.shapes.append(shape)
