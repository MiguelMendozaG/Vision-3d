import cv2
import numpy as np
import matplotlib.pyplot as plt


x = [17.3,
19.3,
19.5,
19.7,
22.9,
23.1,
26.4,
26.8,
27.6,
28.1,
28.2,
28.7,
29,
29.6,
29.9,
29.9,
30.3,
31.3,
36,
39.5,
40.4,
44.3,
44.6,
50.4,
55.9]

y = [71.7,
48.3,
88.3,
75,
91.7,
100,
73.3,
65,
75,
88.3,
68.3,
96.7,
76.7,
78.3,
60,
71.7,
85,
85,
88.3,
100,
100,
100,
91.7,
100,
71.7]

def mu(x2, b1, b0):
	y2 = b0 +b1*x2
	return y2

xi = 0
yi = 0
xi_yi = 0
xi_xi = 0
n = len(x)
y_gorro = 0
y_sal = []
residual = []

for a in range(0,len(x)):
	xi += x[a]
	yi += y[a]
	xi_yi +=  x[a]*y[a]
	xi_xi+= x[a]*x[a]

b1 = (n*xi_yi-xi*yi)/(n*xi_xi-xi*xi)
b0 = (yi-b1*xi)/n

mu_y_30 = mu (30,b1,b0)

print('b1',b1)
print('b0',b0)
print('mu',mu_y_30)
a = 0
	
for a in range(0,len(x)):
	y_gorro = mu(x[a],b1,b0)
	y_sal.append(y_gorro)
	residual.append(y[a] - y_gorro)

print(y)

#plt.plot(x,y)
#plt.show()
plt.ioff()
plt.plot(x,residual)
#plt.plot(x,y_sal)
plt.show()
plt.ion()



	



