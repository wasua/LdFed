import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import models
from torchvision import transforms
import construct_models as cm
from statistics import NormalDist
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor as LOF
import time

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)

def train_with_clipping(model, train_loader, criterion, optimizer, device, clipping=True, clipping_threshold=10):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if clipping:
            torch.nn.utils.clip_grad_value_(model.parameters(), clipping_threshold)
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)

def train_with_augmentation(model, train_loader, criterion, optimizer, device, clipping, clipping_threshold=10, use_augmentation=False, augment=None ):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        if use_augmentation:
            data = augment(data)

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if clipping:
            torch.nn.utils.clip_grad_value_(model.parameters(), clipping_threshold)

        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)

def validation(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)

    return (test_loss, correct)

def get_model_names(model_dict):
    name_of_models = list(model_dict.keys())
    return name_of_models

def get_optimizer_names(optimizer_dict):
    name_of_optimizers = list(optimizer_dict.keys())
    return name_of_optimizers

def get_criterion_names(criterion_dict):
    name_of_criterions = list(criterion_dict.keys())
    return name_of_criterions

def get_x_train_sets_names(x_train_dict):
    name_of_x_train_sets = list(x_train_dict.keys())
    return name_of_x_train_sets

def get_y_train_sets_names(y_train_dict):
    name_of_y_train_sets = list(y_train_dict.keys())
    return name_of_y_train_sets

def get_x_valid_sets_names(x_valid_dict):
    name_of_x_valid_sets = list(x_valid_dict.keys())
    return name_of_x_valid_sets

def get_y_valid_sets_names(y_valid_dict):
    name_of_y_valid_sets = list(y_valid_dict.keys())
    return name_of_y_valid_sets

def get_x_test_sets_names(x_test_dict):
    name_of_x_test_sets = list(x_test_dict.keys())
    return name_of_x_test_sets

def get_y_test_sets_names(y_test_dict):
    name_of_y_test_sets = list(y_test_dict.keys())
    return name_of_y_test_sets

def create_model_optimizer_criterion_dict_for_mnist(number_of_samples, learning_rate, momentum, device, is_cnn=False, weight_decay=0):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)
        if is_cnn:
            model_info = cm.Netcnn()
        else:
            model_info = cm.Net2nn()
        model_info = model_info.to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def create_model_optimizer_criterion_dict_for_cifar_net(number_of_samples, learning_rate, momentum, device, weight_decay=0):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)

        model_info = cm.Netcnn_cifar()

        model_info = model_info.to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict

def create_model_optimizer_criterion_dict_for_cifar_cnn(number_of_samples, learning_rate, momentum, device, weight_decay=0):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)

        model_info = cm.Cifar10CNN()

        model_info = model_info.to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum,
                                         weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict

def create_model_optimizer_criterion_dict_for_fashion_mnist(number_of_samples, learning_rate, momentum, device, weight_decay=0):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)

        model_info = cm.Net_fashion()
        model_info = model_info.to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def create_model_optimizer_criterion_dict_for_cifar_resnet(number_of_samples, learning_rate, momentum, device, weight_decay=0):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)

        model_info = models.resnet18(num_classes=10)

        model_info = model_info.to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum,
                                         weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict

"""
distribute the global model to participates
participates update model
"""
def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples):
    name_of_models = list(model_dict.keys())
    main_model_param_data_list = list(main_model.parameters()) # useing model.parameters() to get models' parameters
    with torch.no_grad():# 该句使得模型不会被反向传播更新
        for i in range(number_of_samples):
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters())
            for j in range(len(main_model_param_data_list)):
                sample_param_data_list[j].data = main_model_param_data_list[j].data.clone()
    return model_dict


def compare_local_and_merged_model_performance(number_of_samples, x_test_dict, y_test_dict, batch_size, model_dict, criterion_dict, main_model, main_criterion, device):
    accuracy_table = pd.DataFrame(data=np.zeros((number_of_samples, 3)), columns=["sample", "local_ind_model", "merged_main_model"])

    name_of_x_test_sets = list(x_test_dict.keys())
    name_of_y_test_sets = list(y_test_dict.keys())

    name_of_models = list(model_dict.keys())
    name_of_criterions = list(criterion_dict.keys())

    for i in range(number_of_samples):
        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]

        individual_loss, individual_accuracy = validation(model, test_dl, criterion, device)
        main_loss, main_accuracy = validation(main_model, test_dl, main_criterion, device)

        accuracy_table.loc[i, "sample"] = "sample " + str(i)
        accuracy_table.loc[i, "local_ind_model"] = individual_accuracy
        accuracy_table.loc[i, "merged_main_model"] = main_accuracy

    return accuracy_table



""""
###
###标签翻转攻击训练
###
"""
def start_train_end_node_process_without_print(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                               batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch,
                                               device):
    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)
    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)
    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)

    for i in range(number_of_samples):

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]

        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            #print("--model: ",i,"--epoch: ",epoch)
            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer, device)
            test_loss, test_accuracy = validation(model, test_dl, criterion, device)

def start_train_end_node_process_with_clipping(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                               batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch,
                                               device, clipping=True, clipping_threshold=10):
    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)
    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)
    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)

    for i in range(number_of_samples):

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]

        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train_with_clipping(model, train_dl, criterion, optimizer, device, clipping, clipping_threshold)

            test_loss, test_accuracy = validation(model, test_dl, criterion, device)


"""
###
### cifar的拜占庭攻击训练
###
"""
def start_train_end_node_process_cifar(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                       batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch,
                                       device,clipping=False, clipping_threshold =10):
    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)
    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)
    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)

    transform_augment = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), padding=4)])

    for i in range(number_of_samples):

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train_with_augmentation(model, train_dl, criterion, optimizer, device,
                                                                 clipping=clipping,
                                                                 clipping_threshold=clipping_threshold,
                                                                 use_augmentation=True, augment=transform_augment)

            test_loss, test_accuracy = validation(model, test_dl, criterion, device)


##########################################
"""
###
###带参数的cifar 的拜占庭攻击训练
###
"""

def start_train_end_node_process_byzantine_for_cifar_with_augmentation(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                               batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch, byzantine_node_list,
                                            byzantine_mean, byzantine_std, device, clipping=False, clipping_threshold =10, iteration_byzantine_seed=None ):

    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)

    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)

    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)

    #数据增强
    transform_augment = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), padding=4)])



    trusted_nodes=  np.array(list(set(np.arange(number_of_samples)) - set(byzantine_node_list)), dtype=int)

    ## STANDARD LOCAL MODEL TRAİNİNG PROCESS FOR TRUSTED NODES
    for i in trusted_nodes:

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            #print('--model: ',i,"--epoch: ", epoch)
            train_loss, train_accuracy = train_with_augmentation(model, train_dl, criterion, optimizer, device, clipping=clipping, clipping_threshold=clipping_threshold,
                                                                 use_augmentation=True, augment=transform_augment)
            test_loss, test_accuracy = validation(model, test_dl, criterion, device)

    with torch.no_grad():

        for j in byzantine_node_list:

            hostile_node_param_data_list = list(model_dict[name_of_models[j]].parameters())

            for k in range(len(hostile_node_param_data_list)):
                np.random.seed(iteration_byzantine_seed)
                hostile_node_param_data_list[k].data = torch.tensor(np.random.normal(byzantine_mean,byzantine_std, hostile_node_param_data_list[k].data.shape ), dtype=torch.float32, device=device)

            model_dict[name_of_models[j]].float()


###############################################
""""
###
###拜占庭攻击训练
###
"""
def start_train_end_node_process_byzantine(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                               batch_size, model_dict, criterion_dict, optimizer_dict,
                                               numEpoch, byzantine_node_list, byzantine_mean, byzantine_std, device, iteration_byzantine_seed=None ):

    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)

    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)

    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)



    trusted_nodes=  np.array(list(set(np.arange(number_of_samples)) - set(byzantine_node_list)), dtype=int)

    ## STANDARD LOCAL MODEL TRAİNİNG PROCESS FOR TRUSTED NODES
    # ？？？为什么要减去拜占庭节点，没有防御措施的话，不知道拜占庭节点啊？？？
    # 因为拜占庭节点作为攻击，其上传的参数是任意的（后面单独产生）
    # 所以该论文没有针对某种特定的攻击，实用性不强
    for i in trusted_nodes:
        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            #print('--model: ',i,"--epoch: ", epoch)
            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer, device) #使用cnn之后，模型训练很慢

            test_loss, test_accuracy = validation(model, test_dl, criterion, device)


    ### ？？？？？？？？？
    ##为啥不跟正常节点一起训练呢？
    ##拜占庭上传任意参数攻击（使用随机函数生成任意参数）
    with torch.no_grad():
        for j in byzantine_node_list:
            hostile_node_param_data_list = list(model_dict[name_of_models[j]].parameters())

            for k in range(len(hostile_node_param_data_list)):
                np.random.seed(iteration_byzantine_seed) #伟大的随机数种子
                hostile_node_param_data_list[k].data = \
                    torch.tensor(np.random.normal(byzantine_mean,byzantine_std,
                                                  hostile_node_param_data_list[k].data.shape ), dtype=torch.float32, device=device)

            model_dict[name_of_models[j]].float()


################################################

def compare_individual_models_on_only_one_label(model_dict, criterion_dict, x_just_dict, y_just_dict, batch_size,
                                                device):
    columns = ["model_name"]
    label_names = []
    for l in range(10):
        label_names.append("label" + str(l))
        columns.append("label" + str(l))

    accuracy_rec = pd.DataFrame(data=np.zeros([10, 11]), columns=columns)

    # x_just_dict, y_just_dict = create_just_data(x_test, y_test, x_just_name="x_test_just_", y_just_name="y_test_just_")

    name_of_x_test_just_sets = list(x_just_dict.keys())
    name_of_y_test_just_sets = list(y_just_dict.keys())
    name_of_models = list(model_dict.keys())
    name_of_criterions = list(criterion_dict.keys())

    for i in range(len(name_of_models)):
        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]

        accuracy_rec.loc[i, "model_name"] = name_of_models[i]

        for j in range(10):
            x_test_just = x_just_dict[name_of_x_test_just_sets[j]]
            y_test_just = y_just_dict[name_of_y_test_just_sets[j]]

            test_ds_just = TensorDataset(x_test_just, y_test_just)
            test_dl_just = DataLoader(test_ds_just, batch_size=batch_size * 2)

            test_loss, test_accuracy = validation(model, test_dl_just, criterion, device)

            accuracy_rec.loc[i, label_names[j]] = test_accuracy
    #             print( name_of_models[i], ">>" ,j, " tahmin etmesi: {:7.4f}".format(test_accuracy))
    #         print("******************")
    return accuracy_rec
"""
######
###### 平均聚合 模块
######
"""
""""
用于原始联邦学习中，模型聚合时，计算模型的平均参数
"""
def get_averaged_weights_faster(model_dict, device):
    name_of_models = list(model_dict.keys())
    parameters = list(model_dict[name_of_models[0]].named_parameters()) #name_parameters获取模型的参数名称和对应的值
    ##named_parameters layer adını ve datayı tuple olarak dönderiyor
    ##parameters sadece datayı dönderiyor
    count=len(parameters)
    """for i in range(count):
        print(len(parameters[i][0]),len(parameters[i][1]))"""

    weight_dict = dict()
    #获取模型参数名称、数量及形状
    for k in range(len(parameters)):
        name = parameters[k][0] #获取参数名称
        w_shape = list(parameters[k][1].shape) #对应参数维度
        w_shape.insert(0, len(model_dict)) #插入模型字典长度，使维度和所有模型数量保持一致
        weight_info = torch.zeros(w_shape, device=device) #
        weight_dict.update({name: weight_info}) #初始化参数名和对于参数值（0）

    #获取各个参数
    weight_names_list = list(weight_dict.keys()) #获取参数名称
    with torch.no_grad(): #在该模块下，所有计算得出的tensor的requires_grad都自动设置为False，就不会进行自动求导
        for i in range(len(model_dict)): #遍历所有参与者模型
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters()) #获取该参与者模型的所有参数
            for j in range(len(weight_names_list)): #遍历所有参数
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone() #将该模型参数克隆到weight_dict
                #weight_dict包含了所有模型的所有参数
        #计算各个类别参数的均值
        mean_weight_array = []
        for m in range(len(weight_names_list)):
            mean_weight_array.append(torch.mean(weight_dict[weight_names_list[m]], 0))

    return mean_weight_array

"""
经典联邦学习聚合方法
未采取防御措施
平均聚合
"""
def set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, device):
    mean_weight_array = get_averaged_weights_faster(model_dict, device) #获取到所有模型各个参数对应的均值
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = mean_weight_array[j]
    return main_model

"""
######
###### median 模块
######
"""
"""
计算各个参数的中位数
"""
def get_coordinate_wise_median_of_weights(model_dict, device):
    name_of_models = list(model_dict.keys())
    parameters = list(model_dict[name_of_models[0]].named_parameters())
    ##named_parameters layer adını ve datayı tuple olarak dönderiyor
    ##parameters sadece datayı dönderiyor

    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)
        w_shape.insert(0, len(model_dict))
        weight_info = torch.zeros(w_shape, device=device)
        weight_dict.update({name: weight_info})

    weight_names_list = list(weight_dict.keys())
    with torch.no_grad():
        for i in range(len(model_dict)):
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()

        median_weight_array = []
        for m in range(len(weight_names_list)):
            median_weight_array.append(torch.median(weight_dict[weight_names_list[m]], 0).values)

    return median_weight_array

"""
实现median的功能
"""
def set_coordinatewise_med_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, device):
    median_weight_array = get_coordinate_wise_median_of_weights(model_dict, device)
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = median_weight_array[j]
    return main_model

"""
######
###### trimmed 模块
######
"""
def get_trimmed_mean(model_dict, hostile_node_percentage, device):
    name_of_models = list(model_dict.keys())
    parameters = list(model_dict[name_of_models[0]].named_parameters())
    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)
        w_shape.insert(0, len(model_dict))
        weight_info = torch.zeros(w_shape, device=device)
        weight_dict.update({name: weight_info})
    weight_names_list = list(weight_dict.keys())
    with torch.no_grad():
        for i in range(len(model_dict)):
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()
    mean_weight_array = []
    for m in range(len(weight_names_list)):
        layers_from_nodes = weight_dict[weight_names_list[m]]
        #scipy.stats.trim_mean修剪均值
        #hostile_node_percentage表示从排序之后layers_from_nodes.clone().cpu()的两端各剪去这么多
        #例如hostile_node_percentage=0.1,则从排序好的序列两端各剪去10%
        print("692")
        trim_layer_info=stats.trim_mean(layers_from_nodes.clone().cpu(), hostile_node_percentage, axis=0)
        print("694")
        mean_weight_array.append(trim_layer_info)
    return mean_weight_array

""""
实现trimmed算法
"""
def set_trimmed_mean_weights_as_main_model_weights_and_update_main_model(main_model, model_dict,number_of_samples,
                                                                             hostile_node_percentage, device):
    mean_weight_array = get_trimmed_mean(model_dict, hostile_node_percentage, device)
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
                        main_model_param_data_list[j].data = torch.tensor(mean_weight_array[j], dtype=torch.float32, device=device)
    return main_model

"""
######
###### barfed 模块
######
"""

"""
计算欧式距离
"""
def calculate_euclidean_distances(main_model, model_dict):
    calculated_parameter_names = []

    #获取所需要参与计算的参数名称（不计算bias）（fc1.weight，fc2.weight，fc3.weight）
    for parameters in main_model.named_parameters():  ## bias dataları için distance hesaplamıyorum
        if "bias" not in parameters[0]:
            calculated_parameter_names.append(parameters[0])

    columns = ["model"] + calculated_parameter_names
    distances = pd.DataFrame(columns=columns)#以model，fc1.weight，fc2.weight，fc3.weight为列标题的距离表
    model_names = list(model_dict.keys())

    #获取全局模型的参数名称及对应参数（包含bias）
    main_model_weight_dict = {}
    for parameter in main_model.named_parameters():
        name = parameter[0]
        weight_info = parameter[1]
        main_model_weight_dict.update({name: weight_info})

    with torch.no_grad():
        #遍历所有模型
        for i in range(len(model_names)):
            distances.loc[i, "model"] = model_names[i] #添加对应的模型名称到对应序号
            sample_node_parameter_list = list(model_dict[model_names[i]].named_parameters()) #获取模型参数名及对应参数值
            for j in sample_node_parameter_list:#便利参数名和对应值
                if j[0] in calculated_parameter_names:#
                    distances.loc[i, j[0]] = round( #round四舍五入  ##各个模型和上一轮全局模型计算求距离
                        np.linalg.norm(main_model_weight_dict[j[0]].cpu().data - j[1].cpu().data), 4) #np.linalg.norm求范数

    return distances

"""
计算距离的上下限
"""
def calculate_lower_and_upper_limit(data, factor):
    #pandas.quantile计算分位数
    #计算四分位数
    quantiles = data.quantile(q=[0.25, 0.50, 0.75]).values
    q1 = quantiles[0] #25%位点
    q2 = quantiles[1] #50%位点
    q3 = quantiles[2] #75%位点
    iqr = q3 - q1
    lower_limit = q1 - factor * iqr
    upper_limit = q3 + factor * iqr
    return lower_limit, upper_limit

def get_outlier_situation_and_thresholds_for_layers(distances, factor=1.5):
    layers = list(distances.columns)
    layers.remove("model")
    threshold_columns = []
    for layer in layers:#为每个参数制作上下限表头
        threshold_columns.append((layer + "_lower"))
        threshold_columns.append((layer + "_upper"))
    thresholds = pd.DataFrame(columns=threshold_columns)#每个参数上下限表

    include_calculation_result = True
    for layer in layers:#遍历所有的参数类型
        data = distances[layer] #获取每个参数类型的所有模型距离
        lower, upper = calculate_lower_and_upper_limit(data, factor) #得出每类参数的距离上下限制
        lower_name = layer + "_lower"
        upper_name = layer + "_upper"
        thresholds.loc[0, lower_name] = lower
        thresholds.loc[0, upper_name] = upper #填充参数距离上下限表

        name = layer + "_is_in_ci" #层参数是否合规标志位
        distances[name] = (distances[layer] > lower) & (distances[layer] < upper) #层参数是否在上下限内
        include_calculation_result = include_calculation_result & distances[name] #这个是干啥呀

    distances["include_calculation"] = include_calculation_result #
    return distances, thresholds

"""
根据距离表中是否可信标志筛选出可靠客户端
然后计算苛刻客户端的参数均值
"""
def get_averaged_weights_without_outliers_strict_condition(model_dict, iteration_distance, device):
    #筛选可靠客户
    chosen_clients = iteration_distance[iteration_distance["include_calculation"] == True].index
    name_of_models = list(model_dict.keys())
    parameters = list(model_dict[name_of_models[0]].named_parameters())

    ### mesela conv 1 için zeros [chosen client kadar, 32, 1, 5, 5] atanıyor bunları doldurup mean alacağız
    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)
        w_shape.insert(0, len(chosen_clients))
        weight_info = torch.zeros(w_shape, device=device)
        weight_dict.update({name: weight_info})

    weight_names_list = list(weight_dict.keys())
    with torch.no_grad():
        for i in range(len(chosen_clients)):
            sample_param_data_list = list(model_dict[name_of_models[chosen_clients[i]]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()

        mean_weight_array = []
        for m in range(len(weight_names_list)):
            mean_weight_array.append(torch.mean(weight_dict[weight_names_list[m]], 0))

    return mean_weight_array

""""
实现barfed功能
"""
def strict_condition_without_outliers_set_averaged_weights_as_main_model_weights_and_update_main_model(main_model,
                                                                                                       model_dict,
                                                                                                       iteration_distance,
                                                                                                       device):
    mean_weight_array = get_averaged_weights_without_outliers_strict_condition(model_dict, iteration_distance, device)
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = mean_weight_array[j]
    return main_model


"""
 #######
 #######未知用途代码
 #######
"""


## they do not perform any training and send same parameters that are received at the beginning at the fl round
def start_train_end_node_process_with_anticatalysts(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                                    batch_size, model_dict, criterion_dict, optimizer_dict,
                                                    numEpoch, byzantine_node_list, device):
    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)
    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)
    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)



    trusted_nodes=  np.array(list(set(np.arange(number_of_samples)) - set(byzantine_node_list)), dtype=int)

    ## STANDARD LOCAL MODEL TRAİNİNG PROCESS FOR TRUSTED NODES
    for i in trusted_nodes:

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer, device)
            test_loss, test_accuracy = validation(model, test_dl, criterion, device)


######################################
#### a little is enough functions
def get_zscore_for_a_little_is_enough (number_of_samples,hostile_node_percentage):
#     from statistics import NormalDist is defined at the top
    malicious = int(number_of_samples * hostile_node_percentage)
    supporter = np.floor((number_of_samples / 2) + 1) - malicious
    area = (number_of_samples - malicious - supporter) / (number_of_samples - malicious)
    zscore = NormalDist().inv_cdf(area)
    return zscore

def get_byzantine_node_stats_for_a_little(model_dict,byzantine_node_list, device):
    name_of_models = []
    for node in byzantine_node_list:
        name_of_models.append("model"+str(node))
    parameters = list(model_dict[name_of_models[0]].named_parameters())
    ##named_parameters layer adını ve datayı tuple olarak dönderiyor
    ##parameters sadece datayı dönderiyor
    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)
        w_shape.insert(0, len(byzantine_node_list))
        weight_info = torch.zeros(w_shape, device=device)
        weight_dict.update({name: weight_info})
    weight_names_list = list(weight_dict.keys())
    with torch.no_grad():
        for i in range(len(byzantine_node_list)):
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()
        mean_weight_array = []
        std_weight_array = []
        for m in range(len(weight_names_list)):
            mean_weight_array.append(torch.mean(weight_dict[weight_names_list[m]], 0))
            std_weight_array.append(torch.std(weight_dict[weight_names_list[m]], 0))
    return mean_weight_array,std_weight_array

def change_parameters_of_hostile_nodes(model_dict, byzantine_node_list,zscore, device):
    name_of_models = list(model_dict.keys())
    with torch.no_grad():
        mean_weight_array,std_weight_array = get_byzantine_node_stats_for_a_little(model_dict,byzantine_node_list, device=device)
        for j in byzantine_node_list:
            hostile_node_param_data_list = list(model_dict[name_of_models[j]].parameters())
            for k in range(len(hostile_node_param_data_list)):
                hostile_node_param_data_list[k].data= mean_weight_array[k]-std_weight_array[k]*zscore
            model_dict[name_of_models[j]].float()
    return model_dict

######################################
# fang partial knowledge attack adaptation
def partial_knowledge_fang_ind(main_model, model_dict,byzantine_node_list, iteration_byzantine_seed, device):
    name_of_models = list(model_dict.keys())
    with torch.no_grad():
        mean_weight_array, std_weight_array = get_byzantine_node_stats_for_a_little(model_dict, byzantine_node_list,
                                                                                       device=device)
        main_model_param_data_list = list(main_model.parameters())
        for j in byzantine_node_list:
            hostile_node_param_data_list = list(model_dict[name_of_models[j]].parameters())
            for k in range(len(hostile_node_param_data_list)):
                original_shape = list(hostile_node_param_data_list[k].data.shape)
                data = np.zeros(original_shape)
                hostile_data = hostile_node_param_data_list[k].data.clone().data.cpu()
                main_model_data = main_model_param_data_list[k].data.clone().data.cpu()
                mean = mean_weight_array[k].clone().data.cpu()
                std = std_weight_array[k].clone().data.cpu()
                sign_matrix = (hostile_data > main_model_data)
                np.random.seed(iteration_byzantine_seed) ## nodelar kendi mean stdlerine göre alıyor her experimentin x. roundı aynı gibi
                data[sign_matrix == True] = np.random.uniform(
                    low=mean[sign_matrix == True] - 4 * std[sign_matrix == True],
                    high=mean[sign_matrix == True] - 3 * std[sign_matrix == True])
                np.random.seed(iteration_byzantine_seed)
                data[sign_matrix == False] = np.random.uniform(
                    low=mean[sign_matrix == False] + 3 * std[sign_matrix == False],
                    high=mean[sign_matrix == False] + 4 * std[sign_matrix == False])
                hostile_node_param_data_list[k].data = torch.tensor(data,dtype=torch.float32, device=device)
            model_dict[name_of_models[j]].float()
    return model_dict

def partial_knowledge_fang_org(main_model, model_dict, byzantine_node_list, iteration_byzantine_seed, device):
    name_of_models = list(model_dict.keys())
    with torch.no_grad():
        mean_weight_array, std_weight_array = get_byzantine_node_stats_for_a_little(model_dict, byzantine_node_list,
                                                                                       device=device)
        main_model_param_data_list = list(main_model.parameters())
        organized = []
        for k in range(len(main_model_param_data_list)):
            original_shape = list(main_model_param_data_list[k].data.shape)
            data = np.zeros(original_shape)
            main_model_data = main_model_param_data_list[k].clone().data.cpu()
            mean = mean_weight_array[k].clone().data.cpu()
            std = std_weight_array[k].clone().data.cpu()
            sign_matrix = (mean > main_model_data)
            np.random.seed(iteration_byzantine_seed)
            data[sign_matrix == True] = np.random.uniform(
                low=mean[sign_matrix == True] - 4 * std[sign_matrix == True],
                high=mean[sign_matrix == True] - 3 * std[sign_matrix == True])
            np.random.seed(iteration_byzantine_seed)
            data[sign_matrix == False] = np.random.uniform(
                low=mean[sign_matrix == False] + 3 * std[sign_matrix == False],
                high=mean[sign_matrix == False] + 4 * std[sign_matrix == False])
            organized.append(data)
    for b in byzantine_node_list:
        hostile_node_param_data_list = list(model_dict[name_of_models[b]].parameters())
        for m in range(len(hostile_node_param_data_list)):
            hostile_node_param_data_list[m].data = torch.tensor(organized[m], dtype=torch.float32, device=device)
        model_dict[name_of_models[b]].float()
    return model_dict



"""
####
####  loFed 模块
####
"""

"""
###将各个节点模型分类：正常节点和恶意节点
"""
"""def getParameterFrameAndHostileNode(main_model, model_dict,neighbors):
    parameterList = []
    #获取所需要参与计算的参数名称（不计算bias）（fc1.weight，fc2.weight，fc3.weight）
    for parameters in main_model.named_parameters():  ## bias dataları için distance hesaplamıyorum
        if "bias" not in parameters[0]:
            parameterList.append(parameters[0])

    columnsList = ["model"] + parameterList

    lofFrame = pd.DataFrame(columns=columnsList)#以model，fc1.weight，fc2.weight，fc3.weight为列标题的距离表
    model_names = list(model_dict.keys())
    
    with torch.no_grad():
        #遍历所有模型
        for i in range(len(model_names)):
            lofFrame.loc[i, "model"] = model_names[i] #添加对应的模型名称到对应序号


            sample_node_parameter_list = list(model_dict[model_names[i]].named_parameters()) #获取模型参数名及对应参数值
            for j in sample_node_parameter_list:#遍历参数名和对应值
                if j[0] in parameterList:#
                    size=(j[1].cpu().data).size()
                    lofFrame.loc[i, j[0]] =[i for i in ((j[1].cpu().data).numpy()).flat]
    parameterLofList=[]
    for parameter in parameterList:
        parameterFlagName=parameter+'_Lof' #设置该参数Lof标志列名
        parameterLofList.append(parameterFlagName)
        allParameterValue=lofFrame[parameter] #获取该列（该参数）所有值
        parameterValueList=[value for value in allParameterValue] #需要想办法进行降维
        lof=LOF(n_neighbors=neighbors)#测试
        #lof=LOF(n_neighbors=neighbors,novelty=True) #只接受二维数据
        lofFlag=lof.fit_predict(parameterValueList)# 获取该列的LOF值
        #lofFlag=lof.decision_function(parameterValueList)
        lofFrame[parameterFlagName]=lofFlag #参数Lof标志列赋值

    #判断恶意节点，并对恶意节点标志位赋值
    hostileNodeFlag=True
    for parameterLof in parameterLofList:
        hostileNodeFlag=lofFrame[parameterLof]==-1
    lofFrame["hostileNodeFlag"]=hostileNodeFlag

    hostileNodeList=list(lofFrame.loc[lofFrame["hostileNodeFlag"]==True,].index) #获取恶意节点
    normalNodeList = list(lofFrame.loc[lofFrame["hostileNodeFlag"] == False,].index) #获取正常节点
    print("---predict byznode:",hostileNodeList)
    return normalNodeList,hostileNodeList"""

def getParameterFrameAndHostileNode(main_model, model_dict,neighbors):
    parameterList = []
    #获取所需要参与计算的参数名称（不计算bias）（fc1.weight，fc2.weight，fc3.weight）
    for parameters in main_model.named_parameters():  ## bias dataları için distance hesaplamıyorum
        #if "bias" not in parameters[0] and "conv" not in parameters[0]: # 不考虑偏置和卷积层
        if "bias" not in parameters[0]:
            parameterList.append(parameters[0])

    columnsList = ["model"] + parameterList

    lofFrame = pd.DataFrame(columns=columnsList)  # 以model，fc1.weight，fc2.weight，fc3.weight为列标题的距离表
    model_names = list(model_dict.keys())

    with torch.no_grad():
        # 遍历所有模型
        for i in range(len(model_names)):
            lofFrame.loc[i, "model"] = model_names[i]  # 添加对应的模型名称到对应序号

            sample_node_parameter_list = list(model_dict[model_names[i]].named_parameters())  # 获取模型参数名及对应参数值
            for j in sample_node_parameter_list:  # 遍历参数名和对应值
                if j[0] in parameterList:  #
                    if 'conv' not in j[0]:
                        lofFrame.loc[i, j[0]] = [i for i in ((j[1].cpu().data).numpy())]
                    else:
                        lofFrame.loc[i, j[0]] = [i for i in ((j[1].cpu().data).numpy()).flat]
                        print(type(lofFrame.loc[i, j[0]]))
    parameterLofList=[]
    for parameter in parameterList:
        parameterFlagName=parameter+'_Lof' #设置该参数Lof标志列名
        parameterLofList.append(parameterFlagName)

        lofFrame[parameterFlagName] =0

        allParameterValue=lofFrame[parameter] #获取该列（该参数）所有值
        parameterValueList=[value for value in allParameterValue]
        if 'conv' not in parameter:#使用if来区分只考虑线性层还是得考虑卷积层vi没
            for i in range(len(parameterValueList[0])):
                data=[value[i] for value in parameterValueList]

                lof=LOF(n_neighbors=neighbors)#测试
                #lof=LOF(n_neighbors=neighbors,novelty=True) #只接受二维数据
                lofFlag=lof.fit_predict(data)# 获取该列的LOF值
                #lofFlag=lof.negative_outlier_factor_
                #lofFlag=lof.decision_function(parameterValue
                lofFrame[parameterFlagName] =lofFrame[parameterFlagName]+ lofFlag  # 参数Lof标志列赋值
        else:
            data=parameterValueList
            lof=LOF(n_neighbors=neighbors)
            lofFlag=lof.fit_predict(data)
            lofFrame[parameterFlagName]=lofFrame[parameterFlagName]+lofFlag

    allFlag=0
    for parameter in parameterList:
        length = len(lofFrame[parameter][0])
        allFlag+=length

    allLof=0
    for parameterLof in parameterLofList:
        allLof+=lofFrame[parameterLof]

    lofFrame["hostileNodeScore"]=allLof -allFlag
    hostilNodeFrame=lofFrame.sort_values(by="hostileNodeScore").head(neighbors-1)
    #判断恶意节点，并对恶意节点标志位赋值
    """*****×××该部分代码逻辑存在重大问题*********"""
    #lofFrame["hostileNodeFlag"]=False
    #for parameterLof in parameterLofList:
    #    lof=lofFrame[parameterLof]
    #    hostileNodeFlag=lof<0
    #    lofFrame["hostileNodeFlag"]=lofFrame["hostileNodeFlag"] | hostileNodeFlag

    hostileNodeList=list(hostilNodeFrame['model'])#获取恶意节点
    normalNodeList =[]
    for model in list(set(model_names)-set(hostileNodeList)): #获取正常节点
        normalNodeList.extend(list(lofFrame.loc[lofFrame['model']==model].index))
    hostileNodeListShow=sorted(hostileNodeList)
    return normalNodeList,hostileNodeList

def getParameterMeanForNormalNode(normalNode,model_dict,device):
    name_of_models = list(model_dict.keys())
    parameters = list(model_dict[name_of_models[0]].named_parameters())

    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)
        w_shape.insert(0, len(normalNode))
        weight_info = torch.zeros(w_shape, device=device)
        weight_dict.update({name: weight_info})

    weight_names_list = list(weight_dict.keys())
    with torch.no_grad():
        for i in range(len(normalNode)):
            sample_param_data_list = list(model_dict[name_of_models[normalNode[i]]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()

        meanParameterList = []
        for m in range(len(weight_names_list)):
            meanParameterList.append(torch.mean(weight_dict[weight_names_list[m]], 0))
    return meanParameterList

def loFed(main_model,model_dict, normalNode,device):
    meanWeightList = getParameterMeanForNormalNode(normalNode,model_dict, device)
    mainModelParamDataList = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(mainModelParamDataList)):
            mainModelParamDataList[j].data = meanWeightList[j]
    return main_model


"""
#
#bulyan模块
#
"""
def bulyan(main_model,model_dict,number_of_samples,neighbors,device):
    calculated_parameter_names = []
    for parameters in main_model.named_parameters():  ## bias dataları için distance hesaplamıyorum
        if "bias" not in parameters[0]:
            calculated_parameter_names.append(parameters[0])

    # 获取所有模型的参数名称及对应参数（包含bias）
    model_weight_dict = {}
    for model in model_dict.keys():
        weight_info={}
        for parameter in model_dict[model].named_parameters():
            weight_info[parameter[0]] = parameter[1]
        model_weight_dict.update({model: weight_info})

    columns = ["model","score"]
    krumScore = pd.DataFrame(columns=columns)  # 以model，fc1.weight，fc2.weight，fc3.weight为列标题的距离表
    model_names = list(model_dict.keys())

    with torch.no_grad():
        # 遍历所有模型
        for j in range(len(model_names)):
            model1=model_names[j]
            model_names2=copy.deepcopy(model_names)
            model_names2.remove(model1)
            distances =pd.DataFrame(columns=['model2', 'distance'])

            for i in range(len(model_names2)):
                model2=model_names2[i]
                distances.loc[i,'model2']=model2
                distance=0
                for parameter in calculated_parameter_names:
                    distance+=np.linalg.norm(model_weight_dict[model1][parameter].cpu().data-model_weight_dict[model2][parameter].cpu().data)#计算距离
                distances.loc[i,'distance']=distance

            nearbyModel=distances.sort_values(by="distance",ascending=True).head(number_of_samples-neighbors-2)#最接近的
            score=sum(nearbyModel['distance'])#计算距离的和
            krumScore.loc[j,"model"]=model1
            krumScore.loc[j,"score"]=score
    selectModel=list((krumScore.sort_values(by='score',ascending=True).head(number_of_samples+2-2*neighbors))['model']) #选择c个梯度
    
    new_model_dict={}
    for model in selectModel:
        new_model_dict[model]=model_dict[model]
    
    print(neighbors,num_of_samples)

    trimmed_mean=get_trimmed_mean(new_model_dict, (neighbors-1)/number_of_samples, device)
    main_model_param_data_list = list(main_model.parameters())

    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = torch.tensor(trimmed_mean[j], dtype=torch.float32, device=device)
    return main_model
