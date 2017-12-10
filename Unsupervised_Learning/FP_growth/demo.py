# a = {'a':1}
#
# print('a' in a)
returnSet = []
nodeSet = [1,2,3,4]
for i in range(1, len(nodeSet) + 1, 1):
    import itertools

    returnSet.extend(list(itertools.combinations(nodeSet,i)))
print(returnSet)