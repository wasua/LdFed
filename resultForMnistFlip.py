"""
from MNIST import mnist_lf_attack
from MNIST import mnist_lf_median
from MNIST import mnist_lf_trimmed
from MNIST import mnist_lf_arfed
from MNIST import mnist_lf_loFed
"""
import mnist_lf_autoFed
import mnist_lf_bulyan

from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

import pickle
from time import time


def getTestAccuracies(participates, iteration_num,hostile_node_percentage,neighbors,isCnn,isNoniid,isOrganized):

    """accuracyAndDefend1=mnist_lf_attack.mnistLfAttack(participates, iteration_num,hostile_node_percentage,isCnn,isNoniid,isOrganized)
    fileName1="mnist_lf_attack_"+str(participates)+"_"+str(iteration_num)+"_"+str(hostile_node_percentage)+"_"+str(isCnn)+"_"+str(isNoniid)+'_'+str(isOrganized)+".pkl"
    with open(fileName1,'ab') as f:
        pickle.dump(accuracyAndDefend1,f)

    accuracyAndDefend2=mnist_lf_median.mnistLfMedian(participates, iteration_num,hostile_node_percentage,isCnn,isNoniid,isOrganized)
    fileName2 = "mnist_lf_median_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + str(isCnn) + "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName2, 'ab') as f:
        pickle.dump(accuracyAndDefend2, f)

    accuracyAndDefend3=mnist_lf_trimmed.mnistLfTrimmed(participates, iteration_num,hostile_node_percentage,isCnn,isNoniid,isOrganized)
    fileName3 = "mnist_lf_trimmed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + str(isCnn) + "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName3, 'ab') as f:
        pickle.dump(accuracyAndDefend3, f)

    accuracyAndDefend4=mnist_lf_arfed.mnistLfArfed(participates, iteration_num,hostile_node_percentage,isCnn,isNoniid,isOrganized)
    fileName4 = "mnist_lf_barfed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + str(isCnn) + "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName4, 'ab') as f:
        pickle.dump(accuracyAndDefend4, f)

    accuracyAndDefend5 = mnist_lf_loFed.mnistLfLoFed(participates, iteration_num, hostile_node_percentage,neighbors,isCnn,isNoniid,isOrganized)
    fileName5 = "mnist_lf_loFed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + str(isCnn) + "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName5, 'ab') as f:
        pickle.dump(accuracyAndDefend5, f)

    accuracyAndDefend6 = mnist_lf_autoFed.mnistLfAutoFed(participates, iteration_num, hostile_node_percentage,
                                                     isCnn, isNoniid, isOrganized)
    fileName6 = "mnist_lf_AutoFed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + str(isCnn) + "_" + str(isNoniid) + '_' + str(isOrganized) + ".pkl"
    with open(fileName6, 'ab') as f:
        pickle.dump(accuracyAndDefend6, f)
    """
    accuracyAndDefend7 = mnist_lf_bulyan.mnistLfbulyan(participates, iteration_num, hostile_node_percentage,
                                                         neighbors,
                                                         isCnn, isNoniid, isOrganized)
    fileName7 = "mnist_lf_bulyan_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + str(isCnn) + "_" + str(isNoniid) + '_' + str(isOrganized) + ".pkl"
    with open(fileName7, 'ab') as f:
        pickle.dump(accuracyAndDefend7, f)


if __name__=="__main__":
    participates=50
    iteration_num=100
    hostile_node_percentages=[0.1,0.2,0.3,0.4]
    numEpoch=10
    isCnn=True
    isNoniid=[False]
    isOrganized=False
    for isnoniid in isNoniid:
        for hostile_node_percentage in hostile_node_percentages:
            neighbors = int(participates * hostile_node_percentage) + 1
            sourceData=getTestAccuracies(participates,iteration_num,hostile_node_percentage,neighbors,isCnn,isnoniid,isOrganized)

