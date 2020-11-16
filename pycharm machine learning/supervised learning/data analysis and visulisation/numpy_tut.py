import numpy as np
'''
1D array in numpy
'''
a = np.array([1,2,3]) # we can put [[]] for 2D array [[[]]] 3D aray....
print(a)
print(a.shape)
print(a[0],a[1],a[2])
a[0]=5 # to change
print(a)
'''
2D array in numpy
'''
b = np.array([[1,2,3],[4,5,6]])
print(b)
print(b.shape) # 2 rows 3 colums
print(b[1][0]) # there is another format
#print(b[1,0])
c = np.zeros((2,5)) # to do an array filled with 0 2 rows 5 colums
print(c)
m = np.ones((2,4)) # array filled with one
print(m)
j = np.full((2,4),9) # to fill an array with a constant ((size of the array),constant)
print(j)
e = np.random.random((2,4))# to fill a array with numbers between  0 and 1
print(e)
l = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(l)
p = l[:2, 1:3] # :2, means the first 2 colums 1:3 wich row
print(p) # if we modify p it will modify l as well
p[[1],[0]] = 11 # it will modify both p and l
print(p)
print('lmvnekmk')
print(l)
kol = l >2
print(kol) # it will give us boleans wheter the number is greater than 2 or not in the array
print(a[a>2]) # will print for us the items greater than two
x = np.array([[1,2],[3,4]])
y = np.array([[9,2],[10,12]])
print(x+y) # or we can
#print(np.add(x,y))
print(x*y) # or
print(np.multiply(x,y))
print(np.sum(x,axis=0))# some of all the items in x in cloumn we put axis 0 rows axis = 1

