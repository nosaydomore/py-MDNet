#-*- coding:utf8 -*-

import numpy as np

sampleSet = np.array([[5,5],[9,1],[8,2],[4,6],[7,3]],float)
QA_B = np.zeros((5,2),float)

pz1 = 0.5
pz2 = 0.5

thetaA = 0.6
thetaB = 0.5
for cou in range(10):
    print '第', cou+1, '次迭代'
    for j in range(5):
        QA_B[j, 0] = pz1 * pow(thetaA, sampleSet[j, 0]) * pow(1 - thetaA, sampleSet[j, 1])\
                     /(pz1 * pow(thetaA, sampleSet[j, 0]) * pow(1 - thetaA, sampleSet[j, 1])
                       +pz2 * pow(thetaB, sampleSet[j, 0]) * pow(1 - thetaB, sampleSet[j, 1]))
        QA_B[j, 1] = pz2 * pow(thetaB, sampleSet[j, 0]) * pow(1 - thetaB, sampleSet[j, 1]) \
                     / (pz1 * pow(thetaA, sampleSet[j, 0]) * pow(1 - thetaA, sampleSet[j, 1])
                        + pz2 * pow(thetaB, sampleSet[j, 0]) * pow(1 - thetaB, sampleSet[j, 1]))

    print QA_B

    sums = np.zeros(2)
    for j in range(2):
        for i in range(5):
            sums[j] = sums[j] + QA_B[i,j] * sampleSet[i, 0]

    sumFm = np.sum(QA_B,axis=0)
    thetaA = sums[0]/sumFm[0]/10
    thetaB = sums[1]/sumFm[1]/10
    print 'thetaA',thetaA
    print 'thetaB',thetaB
