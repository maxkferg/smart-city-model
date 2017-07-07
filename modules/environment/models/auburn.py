from math import pi
from ..components import visual
from ..components.intersection import TrafficLane, Intersection


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
edge1 = visual.Edgeline((0,300),(250,300))
edge2 = visual.Edgeline((600,300),(800,300))
arc1 = visual.Arc([(200,200),(100,100)],1.5*pi,2*pi)
center1 = visual.Centerline((600,400),(800,400))
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
edge1 = visual.Edgeline((0,600),(250,600))
edge2 = visual.Edgeline((600,600),(800,600))
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
edge1 = visual.Edgeline((300,0),(300,250))
edge2 = visual.Edgeline((300,600),(300,800))
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
edge1 = visual.Edgeline((600,0),(600,250))
edge2 = visual.Edgeline((600,600),(600,800))
center1 = visual.Centerline((500,0),(500,300))
lanes.append(TrafficLane(corners,[edge1,edge2,center1]))


intersection = Intersection(lanes)


