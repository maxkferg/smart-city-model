import components
from math import pi

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
        self.shapes.insert(0,components.Lane(corners))



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



lanes = []

##############################
#          Top x lane        #
##############################
corners = [
    (0,300),
    (0,400),
    (800,400),
    (800,300)
]
edge1 = components.Edgeline((0,300),(250,300)) 
edge2 = components.Edgeline((600,300),(800,300))
arc1 = components.Arc([(200,200),(100,100)],1.5*pi,2*pi)
center1 = components.Centerline((600,400),(800,400)) 
lanes.append(TrafficLane(corners,[edge1,edge2,center1,arc1]))


##############################
#        Middle x lane       #
##############################
corners = [
    (0,400),
    (0,500),
    (800,500),
    (800,400)
]

lanes.append(TrafficLane(corners,[]))

##############################
#        Bottom x lane       #
##############################
corners = [
    (0,500),
    (0,600),
    (800,600),
    (800,500)
]
edge1 = components.Edgeline((0,600),(250,600)) 
edge2 = components.Edgeline((600,600),(800,600)) 
lanes.append(TrafficLane(corners,[edge1,edge2]))


##############################
#          Left y lane       #
##############################
corners = [
    (300,0),
    (400,0),
    (400,800),
    (300,800)
]
edge1 = components.Edgeline((300,0),(300,250)) 
edge2 = components.Edgeline((300,600),(300,800)) 
lanes.append(TrafficLane(corners,[edge1,edge2]))


##############################
#        Middle y lane       #
##############################
corners = [
    (400,0),
    (500,0),
    (500,800),
    (400,800)
]

lanes.append(TrafficLane(corners,[]))

##############################
#        Right x lane        #
##############################
corners = [
    (500,0),
    (600,0),
    (600,800),
    (500,800)
]
edge1 = components.Edgeline((600,0),(600,250)) 
edge2 = components.Edgeline((600,600),(600,800)) 
center1 = components.Centerline((500,0),(500,300)) 
lanes.append(TrafficLane(corners,[edge1,edge2,center1]))


intersection = Intersection(lanes)


