#polynomial fitting fom https://www.youtube.com/watch?v=0TSvo2hOKo0
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.polynomial import poly
np.random.seed(1)

x=np.array([1,2,3,4,5])
y=np.array([3,2,0,4,5])
plt.plot(x,y,'o', label='Data Points')

p1=np.polyfit(x,y,deg=1)
y_fit1=np.polyval(p1,x)
new_y=np.poly1d(p1)
new_x=np.linspace(0,6,60)
plt.plot(new_x,new_y(new_x), label='N=1') 
#plt.plot(x,y_fit1)

p2=np.polyfit(x,y,deg=2)
y_fit2=np.polyval(p2,x)
new_y=np.poly1d(p2)
new_x=np.linspace(0,6,60)
plt.plot(new_x,new_y(new_x), label='N=2')
#plt.plot(x,y_fit2)

p4=np.polyfit(x,y,deg=4)
y_fit4=np.polyval(p4,x)
new_y=np.poly1d(p4)
new_x=np.linspace(0,6,60)
plt.plot(new_x,new_y(new_x), label='N=4')




plt.xlim(0, 6)
plt.ylim(-6, 6.5)
plt.grid()
plt.legend()
plt.show()