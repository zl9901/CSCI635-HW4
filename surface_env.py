from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, pi

x = np.arange(100)
y = np.arange(100)
x = x / 100.0
y = y / 100.0

#title="function"
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')


def relu(X):
   return np.maximum(0,X)


def plot(f):
	ax.clear()
	plt.title(title)
	ax.plot_surface(xcalc(x,y, lambda i,j: i), xcalc(x,y, lambda i,j: j), xcalc(x,y, f))
	return ax

def plotf(d, f): 
	plt.clf()
	plt.plot(d, [f(x) for x in d])

def xcalc(x, y, f):
	return np.array([[f(i,j) for i in y] for j in x])

def cube_at(x,y,z,ax,width=.1,color="green"):
	cubies = [0,0,width,width]
	ex = xcalc(cubies, cubies, lambda i,j: i) + x - width/2
	ey = xcalc(cubies, cubies, lambda i,j: j) + y  - width/2
	ez = np.array([[0,0,0,0],[0,width,width,0],[0,width,width,0],[0,0,0,0]]) + z  - width/2
	ax.plot_surface(ex,ey,ez,color=color)
	return ex, ey, ez

def xor_cubes():
	cube_at(0,0,0,ax,color="red")
	cube_at(1,1,0,ax,color="red")
	cube_at(0,1,1,ax,color="green")
	cube_at(1,0,1,ax,color="green")

def xor_clear():
	ax.clear()
	xor_cubes()
	plt.title("XOR")

def xor_plot(f):
	xor_clear()
	ax.plot_surface(xcalc(x,y, lambda i,j: i), xcalc(x,y, lambda i,j: j), xcalc(x,y, f))

def square_at(a,b,width=.05,t=0):
	return lambda i,j: relu(-relu(i*cos(t)+j*sin(t)-a-width) -relu(i*cos(t+pi)+j*sin(t+pi)+a-width) - relu(j*sin(t+pi/2)+i*cos(t+pi/2)-b-width) - relu(j*sin(t+3*pi/2)+i*cos(t+3*pi/2)+b-width) + .2)

def plus(f,g):
	return lambda i,j: f(i,j) + g(i,j)

def reduce(f, a):
	if len(a) == 0:
		return None
	elif len(a) == 1:
		return a[0]
	else:
		return f(a[0], reduce(f,a[1:]))



