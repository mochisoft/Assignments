#Implementing Gradient Descent Algorithm
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = [(2*val +5) for val in x]
# y = 2x+5
y = np.array([7, 9, 11, 13, 15, 17, 19, 21, 23, 25]) 
# The goal is to use gradient descent to generate the value of m and c from y = 2x + 5 (m=2, c=5)
def gradient_descent(x,y):
    # set the start , and c values to 0
    m_cur, c_cur = 0, 0
    n = len(x)
    # The work of a data scientist is to play with the iterations and learning rate
    # until the most optimum values are achieved
    iterations = 10000    
    learning_rate = 0.008
    
    for i in range(iterations):
        
        y_predicted = m_cur*x + c_cur
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)]) # MSE = (1/n) * sum(actual - predicted)**2, think of variance
        dm = -(2/n)*sum(x*(y-y_predicted)) #
        dc = -(2/n)*sum(y-y_predicted)

        m_cur = m_cur - learning_rate * dm # remember, m= m-learning rate * d/dm
        c_cur = c_cur -learning_rate * dc # remember, c= c-learning rate * d/dc
        print(f"m= {m_cur}, c= {c_cur},cost= {cost}, iteration= {i}")

gradient_descent(x,y)    
