import random as r
import math

def app(a, b):
    a.append(b)

''' 
Sort tuple in the provided array in increasing order according
to their distance to D_t
Parameters
----------
array :
        array of tuple of the form [distance from D_t, label, id]
'''
def sort(array):
    s = []
    while (len(array) > 0):
        mini = array[0]
        index = 0
        for i in range(len(array)):
            if (mini[0] >= array[i][0]):
                mini = array[i]
                index = i
        s.append(array[index])
        del array[index]
    return s
   
''' 
Count the number of tuple in tupleSet closer to D_t than refTuple
Parameters
----------
tupleSet :  
            array of tuple of the form [distance from D_t, label, id]
refTuple :  [double, integer, integer]
mode :      0 or 1
            If 0: return the number of tuple as specified
            above but only considering tuple with L_t
            as label
            If 1: return the number of tuple as specified
            above but only considering tuple with a
            different label than L_t
'''
def count(tupleSet, refTuple, mode, L_t):
    j = 0
    if mode==0:
        for i in range(len(tupleSet)):
            if tupleSet[i][0] <= refTuple[0] and tupleSet[i][1] == L_t:
                j=j+1
    elif mode == 1:
        for i in range(len(tupleSet)):
            if tupleSet[i][0] <= refTuple[0] and tupleSet[i][1] != L_t:
                j=j+1
    return j
    
'Compute n choose k'
def choose(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

'''
Compute the number of tuple with label L_t among the
Kth nearest from D_t tuples in the set pipOut
Parameters
----------
pipOut :  
        array of tuple of the form [distance from D_t, label, id]
K :     integer
L_t     integer
'''
def topKWithLt(pipOut, K, L_t):
    j = 0
    for i in range(min(K, len(pipOut))):
        if pipOut[i][1] == L_t:
            j=j+1
    return j

'''
Compute every boolean vector of length n and store them in the
combinations parameter. Should be called with current = [] and
combinations = []
Parameters
----------
n :             integer
                desired length 
current :       boolean vector
combinations :  array of boolean vectors
'''
def allCombi(n, current, combinations):
    if len(current) == n:
        combinations.append(current)
    else:
        allCombi(n, current+[0], combinations)
        allCombi(n, current+[1], combinations)

'''
Take an array of boolean vector as input and filter out
every element that have the q-th element equal to 1
Parameters
----------
array : array of boolean vector
q :     integer smaller then the length of the largest 
        boolean vector in parameter array
'''
def filterQ(array, q):
    temp = []
    for a in array:
        if (a[q-1] == 0):
            temp.append(a)
    return temp

'''
Compute the output of a parameterized pipeline.
Parameters
----------
D :         an array of fork sets, hence an array of array of tuple
vector :    a boolean vector acting as the parameter of the pipeline
'''
def pipelineOutput(D, vector):
    temp = []
    for i in range(len(vector)):
        if vector[i] == 1:
            temp = temp+D[i]
    return sort(temp)

'Return the number of 1s in an integer array'
def countOnes(array):
    summ = 0
    for i in array:
        summ = summ+i
    return summ

'Return a copy of the target vector'
def copyVector(target):
    temp = []
    for i in target:
        temp.append(i)
    return temp

'Return a vector containing the length of each set in D'
def forkSetSize(D):
    sizes = []
    for d in D:
        sizes.append(len(d))
    return sizes

'''
Generate a random data set of the following form: nUnit sets of tuple of the form [distance, label].
The sets are returned in one single array.
Parameters
----------
nUnit :                    integer
                           the desired number of set
lowerBoundForkSetLength :  integer
                           lower bound (included) for the number of tuple in each of the nUnit sets
upperBoundForkSetLength :  integer
                           upper bound (included) for the number of tuple in each of the nUnit sets
distanceLowerBound      :  double
                           lower bound (included) for the distance to D_t
distanceUpperBound      :  double
                           upper bound (included) for the distance to D_t
largest_label           :  integer
                           label will be randomly assigned from 0 to largest_label included
'''
def randomDataSet(nUnit, lowerBoundForkSetLength, upperBoundForkSetLength, distanceLowerBound, distanceUpperBound, largest_label):
    D = []
    for i in range(nUnit):
        fork_set_i = []
        length = r.randint(lowerBoundForkSetLength, upperBoundForkSetLength)
        for j in range(length):
            j_th_tuple = [0, 0]
            j_th_tuple[0] = r.randint(distanceLowerBound, distanceUpperBound)+r.randint(0,1000)*0.001
            j_th_tuple[1] = r.randint(0, largest_label)
            fork_set_i.append(j_th_tuple)
        D.append(fork_set_i)
    return D

'''
Return true if set forkSet contains at least one tuple closer to D_t than D_b or at least
one tuple closer to D_t than D_c
Parameters
----------
forkSet :   set of set of tuple
D_b :       tuple
D_c :       tuple
'''
def tooFarSet(forkSet, D_b, D_c):
    j1 = 0
    for i in range(len(forkSet)):
        if forkSet[i][0] <= D_b[0]:
            j1=j1+1
    j2 = 0
    for i in range(len(forkSet)):
        if forkSet[i][0] <= D_c[0]:
            j2=j2+1
    return (j1 > 0) or (j2 > 0)

'''
Return the tally vectors in tallies updated with the values corresponding to
the fork set Di, the tuples D_c and D_c and the label L_t
Parameters
----------
tallies : set of tally vectors
          tallies[0] corresponds to the tally vector tb
          tallies[1] corresponds to the tally vector sc
          tallies[2] corresponds to the tally vector tbp
          tallies[3] corresponds to the tally vector scp
Di :      the fork set we want to add in the tally vectors
D_b :     a tuple
D_c :     a tuple
L_t :     a label
'''
def appendTally(tallies, Di, D_b, D_c, L_t):
    tempTb = copyVector(tallies[0])
    tempSc = copyVector(tallies[1])
    tempTbp = copyVector(tallies[2])
    tempScp = copyVector(tallies[3])
    tempTb.append(count(Di, D_b, 0, L_t))
    tempTbp.append(count(Di, D_b, 1, L_t))
    tempSc.append(count(Di, D_c, 0, L_t))
    tempScp.append(count(Di, D_c, 1, L_t))
    return [tempTb, tempSc, tempTbp, tempScp]

def minus(v):
    res = []
    for i in range(len(v)):
        res.append((-1)*v[i])
    return res

def sum(v1, v2):
    assert(len(v1) == len(v2))
    res = []
    for i in range(len(v1)):
        res.append(v1[i]+v2[i])
    return res
        



    
