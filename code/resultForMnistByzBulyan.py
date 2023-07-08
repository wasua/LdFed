import fashion_byz_bulyan
import fashion_lf_bulyan
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

import pickle
from time import time


def getTestAccuracies(participates, iteration_num,hostile_node_percentage,neighbors,isNoniid,isOrganized):
    
    accuracyAndDefend1=fashion_byz_bulyan.fashionmnistByzBulyan(participates, iteration_num,hostile_node_percentage,isNoniid,isOrganized,neighbors)
    fileName1 = "fashion_bzy_bulyan_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName1, 'ab') as f:
        pickle.dump(accuracyAndDefend1, f)
        
    accuracyAndDefend2 = fashion_lf_bulyan.fashionmnistFlipBulyan(participates, iteration_num, hostile_node_percentage,
                                                                  isNoniid, isOrganized, neighbors)
    fileName2 = "fashion_lf_bulyan_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + "_" + str(isNoniid) + '_' + str(isOrganized) + ".pkl"
    with open(fileName2, 'ab') as f:
        pickle.dump(accuracyAndDefend1, f)

    return [accuracyAndDefend1,accuracyAndDefend2]

if __name__=="__main__":
    participates=50##40以前准
    iteration_num=100
    hostile_node_percentage=[0.1]#2,0.3,0.4]
    numEpoch=10

    isNoniid=[False]
    isOrganized=False
    #neighbors=5
    for hostile_percentage in hostile_node_percentage:
        for isnoniid in isNoniid:
            t1 = time()
            neighbors = int(participates * hostile_percentage) + 1
            sourceData=getTestAccuracies(participates,iteration_num,hostile_percentage,neighbors,isnoniid,isOrganized)
            t2=time()
            print("   ---------------time:",t2-t1,'---------------')

