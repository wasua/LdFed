import mnist_byz_attack
import mnist_byz_median
import mnist_byz_trimmed
import mnist_byz_arfed
import mnist_byz_loFed
import mnist_byz_bulyan

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

import pickle
from time import time



def getTestAccuracies(participates, iteration_num,hostile_node_percentage,neighbors,isCnn,isNoniid,isOrganized):
    accuracyAndDefend1=mnist_byz_attack.mnistByzAttack(participates, iteration_num,hostile_node_percentage,isCnn,isNoniid,isOrganized)
    fileName1="mnist_bzy_attack_"+str(participates)+"_"+str(iteration_num)+"_"+str(hostile_node_percentage)+"_"+str(isCnn)+"_"+str(isNoniid)+'_'+str(isOrganized)+".pkl"
    with open(fileName1,'ab') as f:
        pickle.dump(accuracyAndDefend1,f)

    accuracyAndDefend2=mnist_byz_median.mnistByzMedian(participates, iteration_num,hostile_node_percentage,isCnn,isNoniid,isOrganized)
    fileName2 = "mnist_bzy_median_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + str(isCnn) + "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName2, 'ab') as f:
        pickle.dump(accuracyAndDefend2, f)

    accuracyAndDefend3=mnist_byz_trimmed.mnistByzTrimmed(participates, iteration_num,hostile_node_percentage,isCnn,isNoniid,isOrganized)
    fileName3 = "mnist_bzy_trimmed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + str(isCnn) + "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName3, 'ab') as f:
        pickle.dump(accuracyAndDefend3, f)

    accuracyAndDefend4=mnist_byz_arfed.mnistByzAfred(participates, iteration_num,hostile_node_percentage,isCnn,isNoniid,isOrganized)
    fileName4 = "mnist_bzy_barfed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + str(isCnn) + "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName4, 'ab') as f:
        pickle.dump(accuracyAndDefend4, f)

    accuracyAndDefend5 = mnist_byz_loFed.mnistByzLofed(participates, iteration_num, hostile_node_percentage,neighbors,isCnn,isNoniid,isOrganized)
    fileName5 = "mnist_bzy_loFed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + str(isCnn) + "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName5, 'ab') as f:
        pickle.dump(accuracyAndDefend5, f)
    accuracyAndDefend6 = mnist_byz_bulyan.mnistByzbulyan(participates, iteration_num, hostile_node_percentage, neighbors, isCnn, isNoniid, isOrganized)
    fileName6 = "mnist_byz_bulyan_" + str(participates) + "_" + str(iteration_num) + "_" + str(
          hostile_node_percentage) + "_" + str(isCnn) + "_" + str(isNoniid) + '_' + str(isOrganized) + ".pkl"
    with open(fileName6, 'ab') as f:
        pickle.dump(accuracyAndDefend6,f)


if __name__=="__main__":
    participates=50 ##40以前准
    iteration_num=100
    hostile_node_percentages=[0]#[0.1,0.2,0.3,0.4]
    numEpoch=10
    """
    随着恶意节点比列增大，neighbors参数应该降低，最好的情况是比恶意节点数量多一
    """

    isCnn=True
    isNoniid=[True,False]
    isOrganized=False
    for isnoniid in isNoniid:
        for hostile_node_percentage in hostile_node_percentages:
            neighbors = int(participates * hostile_node_percentage) + 1 #该数量要比byz节点数量多
            getTestAccuracies(participates,iteration_num,hostile_node_percentage,neighbors,isCnn,isnoniid,isOrganized)
    """
    虽然loFed能够完美识别出恶意节点，但也将恶意节点中正常数据带来的增益一并剔除了，所以导致效果可能不够理想
    """
