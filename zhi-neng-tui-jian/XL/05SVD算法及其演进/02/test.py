# import numpy as np

# print(np.random.random(size=(1,5)))

# a = [1,1,2,3,5,6,8,9,9]
# a = list(set(a))
# print(a)

import numpy as np

a = np.array([1,2,3])
b = np.array([[10,20,30],[1,1,1]])

print(b*10 - a*10)

# err_score = 0.1

# self.P[user] += self.lr * (err_score * self.Q[item] - self.re * self.P[user])
# self.Q[item] += self.lr * (err_score * self.P[user] - self.re * self.Q[item])


print(np.dot(a,b.T))