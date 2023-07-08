import cifar_byz_attack
import cifar_byz_median
import cifar_byz_trimmed
import cifar_byz_barfed
import cifar_byz_loFed
import cifar_byz_bulyan

import pickle
from time import time


def getTestAccuracies(participates, iteration_num,hostile_node_percentage,neighbors,isNoniid,isOrganized):
    accuracyAndDefend1=cifar_byz_attack.cifarByzAttack(participates,iteration_num,hostile_node_percentage,isNoniid,isOrganized)
    fileName1="cifar_bzy_attack_"+str(participates)+"_"+str(iteration_num)+"_"+str(hostile_node_percentage)+"_"+str(isNoniid)+'_'+str(isOrganized)+".pkl"
    with open(fileName1,'ab') as f:
        pickle.dump(accuracyAndDefend1,f)
    accuracyAndDefend2=cifar_byz_median.cifarByzMedian(participates, iteration_num,hostile_node_percentage,isNoniid,isOrganized)
    fileName2 = "cifar_bzy_median_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage)+ "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName2, 'ab') as f:
        pickle.dump(accuracyAndDefend2, f)

    accuracyAndDefend3=cifar_byz_trimmed.cifarByzTrimmed(participates, iteration_num,hostile_node_percentage,isNoniid,isOrganized)
    fileName3 = "cifar_bzy_trimmed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName3, 'ab') as f:
        pickle.dump(accuracyAndDefend3, f)

    accuracyAndDefend4=cifar_byz_barfed.cifarByzBarfed(participates, iteration_num,hostile_node_percentage,isNoniid,isOrganized)
    fileName4 = "cifar_bzy_barfed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage)+ "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName4, 'ab') as f:
        pickle.dump(accuracyAndDefend4, f)
    accuracyAndDefend5 = cifar_byz_loFed.cifarByzLofed(participates, iteration_num, hostile_node_percentage,neighbors,isNoniid,isOrganized)
    fileName5 = "cifar_bzy_loFed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage)+ "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName5, 'ab') as f:
        pickle.dump(accuracyAndDefend5, f)

    accuracyAndDefend6=cifar_byz_bulyan.cifarByzBulyan(participates,iteration_num,hostile_node_percentage,isNoniid,isOrganized,neighbors)
    fileName6 = "cifar_bzy_bulyan_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage)+ "_" + str(isNoniid) + '_' + str(isOrganized) + ".pkl"
    with open(fileName6, 'ab') as f:
        pickle.dump(accuracyAndDefend6, f)

if __name__=="__main__":
    participates=50 ##40以前准
    iteration_num=500
    hostile_node_percentage=[0] ###0代表无攻击
    isNoniid = [True,False]
    numEpoch=10
    isOrganized = True

    for isnoniid in isNoniid:
        for hostile_percentage in hostile_node_percentage:
            t1 = time()
            neighbors = int(participates * hostile_percentage) + 1
            sourceData= getTestAccuracies(participates,iteration_num,hostile_percentage,neighbors,isnoniid,isOrganized)
            t2=time()
            print("   ---------------time:",t2-t1,'---------------')
