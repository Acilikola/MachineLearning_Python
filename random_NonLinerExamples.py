import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

'''
Though Linear regression is very good to solve many problems,
it cannot be used for all datasets.

First recall how linear regression, could model a dataset.
It models a linear relation between a dependent variable y and
independent variable x.

It had a simple equation, of degree 1,
for example y = 2*(x) + 3.
'''
x = np.arange(-5.0, 5.0, 0.1)
##You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
# plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

'''
Non-linear regressions are a relationship between independent variables x
and a dependent variable y which result in a non-linear function
modeled data.

Essentially any relationship that is not linear can be termed
as non-linear, and is usually represented by the polynomial of k degrees
(maximum power of x): y = a*x^3 + b*x^2 + c*x + d

They can have elements like exponentials, logarithms, fractions and others.
For example: y = log(x) or y = log(a*x^3 + ...)
'''

# lets look at: a cubic function graph
x = np.arange(-5.0, 5.0, 0.1)
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# lets look at: a quadratic y = x^2
x = np.arange(-5.0, 5.0, 0.1)
y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# lets look at: an exponential y = a + b*c^x
X = np.arange(-5.0, 5.0, 0.1)
Y= np.exp(X)
plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# lets look at: a logarithmic y = log(x)
X = np.arange(-5.0, 5.0, 0.1)
Y = np.log(X)
plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# lets look at: a sigmoidal/logistic y = a + b / (1 + c^(x-d))
X = np.arange(-5.0, 5.0, 0.1)
Y = 1-4/(1+np.power(3, X-2))
plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()
