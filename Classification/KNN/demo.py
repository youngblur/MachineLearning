# encoding = utf-8

from numpy import *

'''
    Building the array by tuple(元组)
'''
yuanzu = (4,5,6)
one = array(yuanzu)
print(one)

'''
    Building the array by list
'''
list = [1,2,3]
two = array(list)
print(two)

'''
    Building multidimentional array
'''
list1 = [1,2,3]
list2 = [4,5,6]
three = array( [ list1 , list2 ] )
print(three)
print(three[0][2])
print(three[1][2])
print(three*2)

'''
    Building the matrix by tuple list and array
'''
m1 = mat(yuanzu)
print(m1)
m2 = matrix(list)
print(m2)
m3 = mat(three)
print(m3)

'''
    shape : to find the dimension of the matrix
'''
print(m3.shape)
print(m3.shape[0])

'''
    to get the value and  to slice(分片)
'''
print(m3[0,1])
print(m3[0,:])
print(m3[1,0:2]) #[1,0]、[1,1]

'''
    Transposition（转置）
'''
print(m3.T)

'''
    multiply(a,b) 与对应元素相乘
'''

print(multiply(m3,m3))

'''
    tile
'''
print(tile([0,0],5))
print(tile([0,0],(2,1)))

'''
    strip
'''
s = "  a  b "
print(s.strip())

