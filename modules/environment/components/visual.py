from pygame.color import THECOLORS
from pygame.draw import arc, line, polygon


GUIDELINE_COLOR = (0,40,0)


class Line():
    """
    A line which is used for visual purposes
    """
    thickness = 2
    color = THECOLORS["blue"]

    def __init__(self,start,end):
        self.start = start
        self.end = end

    def draw(self,screen):
        line(screen, self.color, self.start, self.end, self.thickness)



class Arc(Line):
    """
    An arc which is used for visual purposes
    """
    thickness = 2
    color = THECOLORS["blue"]
    def __init__(self, rect, start_angle, stop_angle):
        self.rect = rect
        self.start_angle = start_angle
        self.stop_angle = stop_angle

    def draw(self,screen):
        arc(screen, self.color, self.rect, self.start_angle, self.stop_angle, self.thickness)




class Lane(Line):
    """
    A lane which is used for visual purposes
    """
    thickness = 1

    def __init__(self,corners):
        assert len(corners)>=2
        self.corners = corners

    def draw(self,screen):
        polygon(screen, GUIDELINE_COLOR, self.corners, self.thickness)



class Centerline(Line):
    color = THECOLORS["red"]



class Laneline(Line):
    pass


class Edgeline(Line):
    thickness = 2

