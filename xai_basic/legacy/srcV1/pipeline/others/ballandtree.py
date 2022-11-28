import matplotlib.pyplot as plt
import numpy as np


class ObjetOutline2D():
	def __init__(self):
		super(ObjetOutline2D, self).__init__()
		self.s = (512,512)
		self.c = (int(self.s[0]/2),int(self.s[1]/2))

	def get_lattice_coord(self):
		s = self.s
		c = self.c
		xg = np.meshgrid(np.linspace(0,s[0]-1,s[0]).astype(int))
		yg = np.meshgrid(np.linspace(0,s[1]-1,s[1]).astype(int))	
		mx,my = np.meshgrid(xg,yg)	
		x, y = mx - c[0], my - c[1]
		return x, y

	def get_rectangle(self, x, y, bot=0., height=80, half_width=40):
		return (x*0+1.) * (x<=half_width) * (x>=-half_width) * (y>=bot) * (y<=height)

	def get_isosceles(self, x,y, height=40, half_width=10, translation=(0,0)):
		x0,y0 = translation
		m = height/half_width
		left_side = (y-y0 < m*(x+half_width)) * (y-y0>=0.) * (x + half_width < half_width)
		
		right_side = (y-y0>=0)*(x>=0)*(y-y0 < height - m * x)
		isosceles = right_side + left_side
		return isosceles

class Ball(ObjetOutline2D):
	def __init__(self):
		super(Ball, self).__init__()
		
	def get_basic_ball(self, r=24, thickness=3, centerpos=(0,0), preset_color=None,
		RGB=[0.8,0.8,0.8], white_background=False, ):

		x, y = self.get_lattice_coord()
		d = ((x-centerpos[0]) **2+ (y-centerpos[1])**2)**0.5
		ball = (d<=r) * (d>=r-3) 
		
		if preset_color=='yellow':
			ball = np.stack((ball,ball,0*ball))
		else:
			ball = np.stack((RGB[0] * ball, RGB[1] * ball, RGB[2] * ball))		


		return ball.transpose((1,2,0))

class Tree(ObjetOutline2D):
	def __init__(self):
		super(Tree, self).__init__()
		
	def get_basic_tree(self, height=100, bark_rad=10, 
		crown_center=80, crown_height=60, crown_half_width=40,
		tree_RGB=[0.6,0.3,0.15]):
		x, y = self.get_lattice_coord()
		
		# trunk = (x*0+1.) * (x<=bark_rad) * (x>=-bark_rad) * (y>=0.) * (y<=height)
		trunk = self.get_rectangle(x, y, bot=0., height=height, half_width=bark_rad)
		trunk = np.stack((tree_RGB[0]*trunk, tree_RGB[1]*trunk, tree_RGB[2]*trunk))
		trunk = trunk.transpose((1,2,0))
		
		crown = self.get_isosceles(x,y, height=crown_height, half_width=crown_half_width, translation=(0,50))

		return crown 



ba = Ball()
ball = ba.get_basic_ball(r=24, centerpos=(100,100), preset_color=None)
tr = Tree()
tree = tr.get_basic_tree()

fig = plt.figure()
ax = fig.add_subplot(111)
# ax.imshow(ball)
ax.imshow(tree)
ax.set_xlim(0,512)
ax.set_ylim(0,512)
plt.show()