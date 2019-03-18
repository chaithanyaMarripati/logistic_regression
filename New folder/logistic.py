import numpy as np 
import math
import matplotlib.pyplot as plt 
plt.ion()
x=[]
y=[]
n=100
for i in range(n):
	x.append(i)
for i in range(n):
	if i<50:
		y.append(0)
	else:
		y.append(1)
x=np.array([x])
y=np.array([y])

def sig(x):
	return 1/(1+math.exp(-x))
learning_rate=1
m=0
b=0
epochs=500
def m_grad(m,b):
	sum=0
	for i in range(n):
		sum+=(m*(sig(m*x[0,i]+b))-y[0,i])
	return sum/n
def b_grad(m,b):
	sum=0
	for i in range(n):
		sum+=(sig(m*x[0,i]+b) -y[0,i])
	return sum/n 
def run():
	global m,b
	for i in range(epochs):
		plt.clf()
		m=m-(learning_rate*m_grad(m,b))
		b=b-(learning_rate*b_grad(m,b))
		plt.title(f"epoch-{i}")
		a=[-b/m,(-b/m)+1]
		c=[0,m]
		plt.axis([0,70,-1,2])
		plt.plot(a,c)
		plt.scatter(x[0,:50],y[0,:50],color="green")
		plt.scatter(x[0,50:],y[0,50:],color="red")
		plt.draw()
		plt.pause(0.01)
		print(m,b)

run()
print(f"the final values of m and b are {m},{b}")