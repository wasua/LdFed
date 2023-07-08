import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np

import distribute_data as dd
import train_nodes as tn
import construct_models as cm

def mnistByzAttack(participates, iteration_num,hostile_node_percentage,is_cnn,is_noniid,is_organized):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('             ********** attack --', device,' **********')

    number_of_samples = participates # number of participants

    #is_noniid = True #select non-iid

    """
    How many types of data can appear per node
    if noniid,every node can own two types of number
    for example
    node1 only has number 1 and 3
    node2 only has number 4 and 4
    
    if iid ,the node can own all types number 0~9
    """
    if is_noniid:
        n = 3
        min_n_each_node = 3
    else:
        n = 10
        min_n_each_node = 10

    #is_cnn = False #whether is the cnn
    #is_organized = True # whether randomly deal 看各种处理是否是随机的，还是均匀安排的
    #malicious participant ratio

    ##使用mean和std参数作为np.random.normal函数的参数，生成拜占庭节点的任意参数进行攻击
    byzantine_mean =0 #
    byzantine_std =1  #

    iteration_num = iteration_num # number of communication rounds
    learning_rate = 0.01

    weight_decay = 0.0001
    numEpoch = 10
    batch_size = 32
    momentum = 0.9

    seed = 1
    use_seed = 23
    hostility_seed = 88
    converters_seed = 121
    byzantine_seed =25
    factor = 1.5

    train_amount = 4000
    valid_amount = 900
    test_amount = 890
    print_amount = 3

    ######prepare the data
    ## 1 load all data
    x_train, y_train, x_valid, y_valid, x_test, y_test = dd.load_mnist_data() #load all data
    x_test, y_test = dd.get_equal_size_test_data_from_each_label(x_test, y_test, min_amount=test_amount)
    x_train, y_train, x_valid, y_valid, x_test, y_test = map(torch.tensor,(x_train, y_train, x_valid, y_valid, x_test, y_test))
    ## 2 load train data
    label_dict_train = dd.split_and_shuffle_labels(y_data=y_train, seed=seed, amount=train_amount)
    node_label_info_train, total_label_occurences_train, amount_info_table_train = \
        dd.get_info_for_distribute_non_iid_with_different_n_and_amount(
        number_of_samples=number_of_samples, n=n, amount=train_amount, seed=use_seed, min_n_each_node=min_n_each_node)

    x_train_dict, y_train_dict = dd.distribute_mnist_data_to_participants(label_dict=label_dict_train,
                                                                                         amount=train_amount,
                                                                                         number_of_samples=number_of_samples,
                                                                                         n=n, x_data=x_train,
                                                                                         y_data=y_train,
                                                                                         node_label_info=node_label_info_train,
                                                                                         amount_info_table=amount_info_table_train,
                                                                                         x_name="x_train",
                                                                                         y_name="y_train",
                                                                                         is_cnn=is_cnn)
    ### 3 load test data
    label_dict_test = dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount)
    node_label_info_test, total_label_occurences_test, amount_info_table_test = \
        dd.get_info_for_distribute_non_iid_with_different_n_and_amount(
        number_of_samples=number_of_samples, n=n, amount=test_amount, seed=use_seed, min_n_each_node=min_n_each_node)
    x_test_dict, y_test_dict = dd.distribute_mnist_data_to_participants(label_dict=label_dict_test,
                                                                                       amount=test_amount,
                                                                                       number_of_samples=number_of_samples,
                                                                                       n=n, x_data=x_test,
                                                                                       y_data=y_test,
                                                                                       node_label_info=node_label_info_test,
                                                                                       amount_info_table=amount_info_table_test,
                                                                                       x_name="x_test",
                                                                                       y_name="y_test", is_cnn=is_cnn)

    ### 处理数据
    if is_cnn:
        reshape_size = int(np.sqrt(x_train.shape[1])) #np.sqrt取正数平方根
        x_train = x_train.view(-1, 1, reshape_size, reshape_size) #tensor.view改变tensor对象的维度，类似于np.reshape
        x_valid = x_valid.view(-1, 1, reshape_size, reshape_size)
        x_test = x_test.view(-1, 1, reshape_size, reshape_size)

    ##处理 训练数据
    train_ds = TensorDataset(x_train, y_train) #TensorDataset打包数据对象，使其一一对应
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True) #加载划分数据

    ##处理 测试数据
    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

    if is_cnn:
        main_model = cm.Netcnn()
    else:
        main_model = cm.Net2nn()

    main_model = main_model.to(device)
    cm.weights_init(main_model)
    # 构建优化器（随机梯度下降）
    main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    main_criterion = nn.CrossEntropyLoss() #损失函数-交叉损失熵函数
    model_dict, optimizer_dict, criterion_dict = tn.create_model_optimizer_criterion_dict_for_mnist(number_of_samples, learning_rate,
                                                                                                    momentum, device, is_cnn)


    test_accuracies_of_each_iteration = np.array([], dtype=float)
    test_loss_of_each_iteration = []

    #获取拜占庭节点
    byzantine_node_list = dd.get_byzantine_node_list(hostile_node_percentage, number_of_samples, hostility_seed)
    print('---byznode:',byzantine_node_list)

    np.random.seed(byzantine_seed)
    byzantine_seeds_array = np.random.choice(5000, size=iteration_num, replace=False)

    #开始训练
    for iteration in range(iteration_num):
        #分发模型并更新
        model_dict = tn.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples)

        if is_organized:
            iteration_byzantine_seed = byzantine_seeds_array[iteration]
        else:
            iteration_byzantine_seed =None

        #训练局部模型
        tn.start_train_end_node_process_byzantine(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                               batch_size, model_dict, criterion_dict, optimizer_dict,
                                               numEpoch, byzantine_node_list, byzantine_mean, byzantine_std,
                                               device, iteration_byzantine_seed)
        #聚合全局模型
        main_model = tn.set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, device)
        #验证模型准确率
        # ？这个怎么感觉是在全局模型上面验证，不应该实在各个参与者上验证吗？？？？？(个性化之后需要对各个或者各类进行分别验证)
        # 局部模型最后和全局模型一样，所以，只需要对全局模型进行验证即可
        test_loss, test_accuracy = tn.validation(main_model, test_dl, main_criterion, device)
        #收集每轮的准确率
        test_accuracies_of_each_iteration = np.append(test_accuracies_of_each_iteration, test_accuracy)
        test_loss_of_each_iteration.append(test_loss)
        #print(type(test_accuracies_of_each_iteration))
        print("---Iteration", str(iteration + 1), ":accuracy {:7.4f}".format(test_accuracy))
        print("             ", "      loss {:7.4f}".format(test_loss))
    #test_accuracies_of_each_iteration=list(test_accuracies_of_each_iteration)
    #test_accuracies_of_each_iteration.inser(0,0)
    return [list(test_accuracies_of_each_iteration),"attack",test_loss_of_each_iteration]
