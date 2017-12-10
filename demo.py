import numpy as np
a= [ [1,2,3],[4,5,6]]
a = np.mat(a)
print(np.nonzero(a[:,1].A ==2)[0])
# print(a)
# print(np.mean(a,axis=0).tolist()[0])