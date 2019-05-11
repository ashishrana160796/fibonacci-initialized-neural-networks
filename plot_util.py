import matplotlib.pyplot as plt
import pickle
import numpy as np


with open('cost_fib.pkl','rb') as inp:
    cost_gld = pickle.load(inp)
    
with open('cost_rand.pkl','rb') as inp:
    cost_reg = pickle.load(inp)
    
# print(cost_gld)
# print(cost_reg)

x_ax = np.arange(300)

plt.plot(x_ax, cost_gld)
plt.plot(x_ax, cost_reg)
plt.legend(['Cost: Fibonacci Initialization', 'Cost: Random Initialization'], loc='upper right')
plt.xlabel('Epochs', fontsize=13)
plt.ylabel('Cost Function', fontsize=13)
plt.show()


with open('accur_fib.pkl','rb') as inp:
    acc_gld = pickle.load(inp)
    
with open('accur_rand.pkl','rb') as inp:
    acc_reg = pickle.load(inp)
    
print(acc_gld[299]) #97.45
print(acc_reg[299]) #97.88


x_ax = np.arange(300)

plt.plot(x_ax, acc_gld)
plt.plot(x_ax, acc_reg)
plt.legend(['Accuracy: Fibonacci Initialization', 'Accuracy: Random Initialization'], loc='lower right')
plt.xlabel('Epochs', fontsize=13)
plt.ylabel('Percentage Accuracy', fontsize=13)
plt.show()


