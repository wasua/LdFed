import fashion_lf_attack
import fashion_lf_median
import fashion_lf_trimmed
import fashion_lf_barfed
import fashion_lf_loFed
import fashion_lf_bulyan
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

import pickle
from time import time


def getTestAccuracies(participates, iteration_num,hostile_node_percentage,neighbors,isNoniid,isOrganized):
    """
    accuracyAndDefend1=fashion_lf_attack.fashionmnistFlAttack(participates, iteration_num,hostile_node_percentage,isNoniid,isOrganized)
    fileName1="fashionfl_attack_"+str(participates)+"_"+str(iteration_num)+"_"+str(hostile_node_percentage)+"_"+str(isNoniid)+'_'+str(isOrganized)+".pkl"
    with open(fileName1,'ab') as f:
        pickle.dump(accuracyAndDefend1,f)

    accuracyAndDefend2=fashion_lf_median.fashionmnistFlipMedian(participates, iteration_num,hostile_node_percentage,isNoniid,isOrganized)
    fileName2 = "fashion_fl_median_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) +  "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName2, 'ab') as f:
        pickle.dump(accuracyAndDefend2, f)
    accuracyAndDefend3=fashion_lf_trimmed.fashionmnistFlipTrimmed(participates, iteration_num,hostile_node_percentage,isNoniid,isOrganized)
    fileName3 = "fashion_fl_trimmed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage)  + "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName3, 'ab') as f:
        pickle.dump(accuracyAndDefend3, f)
    accuracyAndDefend4=fashion_lf_barfed.fashionmnistFlBarfed(participates, iteration_num,hostile_node_percentage,isNoniid,isOrganized)
    fileName4 = "fashion_fl_barfed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) +  "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName4, 'ab') as f:
        pickle.dump(accuracyAndDefend4, f)

    accuracyAndDefend5 = fashion_lf_loFed.fashionmnistFlipLofed(participates, iteration_num, hostile_node_percentage,neighbors,isNoniid,isOrganized)
    fileName5 = "fashion_fl_loFed_" + str(participates) + "_" + str(iteration_num) + "_" + str(
        hostile_node_percentage) + "_" + str(isNoniid) + '_' + str(isOrganized)+".pkl"
    with open(fileName5, 'ab') as f:
        pickle.dump(accuracyAndDefend5, f)
    """
    accuracyAndDefend7 = fashion_lf_bulyan.fashionmnistFlipBulyan(participates,iteration_num,hostile_node_percentage,isNoniid,isOrganized,neighbors)
    fileName7 = "fashion_lf_bulyan_" + str(participates) + "_" + str(iteration_num) + "_" + str(hostile_node_percentage) + "_" + "_" + str(isNoniid) + '_' + str(isOrganized) + ".pkl"
    with open(fileName7, 'ab') as f:
        pickle.dump(accuracyAndDefend7, f)


if __name__=="__main__":
    participates=50 ##40以前准
    iteration_num=100
    hostile_node_percentage=[0.1]
    numEpoch=10
    """
    随着恶意节点比列增大，neighbors参数应该降低，最好的情况是比恶意节点数量多一
    """

    isNoniid=[True]#,False]
    isOrganized=False
    #neighbors=5
    for isnoniid in isNoniid:
        for hostile_percentage in hostile_node_percentage:
            t1=time()
            neighbors = int(participates * hostile_percentage) + 1 #该数量要比byz节点数量多
            sourceData=getTestAccuracies(participates,iteration_num,hostile_percentage,neighbors,isnoniid,isOrganized)
            t2=time()
            print("   ---------------time:",t2-t1,'---------------')
    """
    虽然loFed能够完美识别出恶意节点，但也将恶意节点中正常数据带来的增益一并剔除了，所以导致效果可能不够理想
    """
