import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np

import distribute_data as dd
import train_nodes as tn
import construct_models as cm

def mnistByzMedian(number_of_samples,iteration_num,hostile_node_percentage,is_cnn,is_noniid,is_organized):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("             ********** median --", device,' **********')

    #number_of_samples = 50 # number of participants

    #is_noniid = True
    if is_noniid:
        n = 3
        min_n_each_node = 3
    else:
        n = 10
        min_n_each_node = 10

    #is_cnn = False
    #is_organized = True
    #hostile_node_percentage = 0.20 #malicious participant ratio
    byzantine_mean =0
    byzantine_std =1

    #iteration_num = 5 # number of communication rounds
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


    x_train, y_train, x_valid, y_valid, x_test, y_test = dd.load_mnist_data()
    x_test, y_test = dd.get_equal_size_test_data_from_each_label(x_test, y_test, min_amount=test_amount)

    x_train, y_train, x_valid, y_valid, x_test, y_test = map(torch.tensor,
                                                             (x_train, y_train, x_valid, y_valid, x_test, y_test))

    ##train
    label_dict_train = dd.split_and_shuffle_labels(y_data=y_train, seed=seed, amount=train_amount)
    node_label_info_train, total_label_occurences_train, amount_info_table_train = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(
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

    ## test
    label_dict_test = dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount)
    node_label_info_test, total_label_occurences_test, amount_info_table_test = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(
        number_of_samples=number_of_samples,
        n=n, amount=test_amount, seed=use_seed, min_n_each_node=min_n_each_node)
    x_test_dict, y_test_dict = dd.distribute_mnist_data_to_participants(label_dict=label_dict_test,
                                                                                       amount=test_amount,
                                                                                       number_of_samples=number_of_samples,
                                                                                       n=n, x_data=x_test,
                                                                                       y_data=y_test,
                                                                                       node_label_info=node_label_info_test,
                                                                                       amount_info_table=amount_info_table_test,
                                                                                       x_name="x_test",
                                                                                       y_name="y_test", is_cnn=is_cnn)

    if is_cnn:
        reshape_size = int(np.sqrt(x_train.shape[1]))
        x_train = x_train.view(-1, 1, reshape_size, reshape_size)
        x_valid = x_valid.view(-1, 1, reshape_size, reshape_size)
        x_test = x_test.view(-1, 1, reshape_size, reshape_size)

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

    if is_cnn:
        main_model = cm.Netcnn()
    else:
        main_model = cm.Net2nn()

    main_model = main_model.to(device)
    cm.weights_init(main_model)

    main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    main_criterion = nn.CrossEntropyLoss()
    model_dict, optimizer_dict, criterion_dict = tn.create_model_optimizer_criterion_dict_for_mnist(number_of_samples, learning_rate,
                                                                                                    momentum, device, is_cnn)


    test_accuracies_of_each_iteration = np.array([], dtype=float)
    test_loss_of_each_iteration = []

    byzantine_node_list = dd.get_byzantine_node_list(hostile_node_percentage, number_of_samples, hostility_seed)
    print('---byznode:',byzantine_node_list)

    np.random.seed(byzantine_seed)
    byzantine_seeds_array = np.random.choice(5000, size=iteration_num, replace=False)


    for iteration in range(iteration_num):

        model_dict = tn.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict,
                                                                       number_of_samples)


        if is_organized:
            iteration_byzantine_seed = byzantine_seeds_array[iteration]
        else:
            iteration_byzantine_seed =None

        tn.start_train_end_node_process_byzantine(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                               batch_size, model_dict, criterion_dict, optimizer_dict,
                                               numEpoch, byzantine_node_list, byzantine_mean, byzantine_std,
                                               device, iteration_byzantine_seed)


        main_model = tn.set_coordinatewise_med_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, device)
        test_loss, test_accuracy = tn.validation(main_model, test_dl, main_criterion, device)

        test_accuracies_of_each_iteration = np.append(test_accuracies_of_each_iteration, test_accuracy)
        test_loss_of_each_iteration.append(test_loss)
        print("---Iteration", str(iteration + 1), ":accuracy {:7.4f}".format(test_accuracy))
        print("             ", "      loss {:7.4f}".format(test_loss))

    return [list(test_accuracies_of_each_iteration),"median",test_loss_of_each_iteration]


