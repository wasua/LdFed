import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import distribute_data as dd
import train_nodes as tn
import construct_models as cm


def fashionmnistByzBulyan(number_of_samples,iteration_num,hostile_node_percentage,is_noniid,is_organized,neighbors):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('             ********** bulyan --', device,' **********')

    #number_of_samples = 2  # number of participants

    #is_noniid = True
    if is_noniid:
        n = 4
        min_n_each_node = 4
    else:
        n = 10
        min_n_each_node = 10

    #is_organized = True
    #hostile_node_percentage = 0.20  # malicious participant ratio
    byzantine_mean = 0
    byzantine_std = 1

    #iteration_num = 2  # number of communication rounds
    learning_rate = 0.0020

    weight_decay = 0.0001
    numEpoch = 10
    batch_size = 25
    momentum = 0.9

    seed = 6
    use_seed = 13
    hostility_seed = 210
    converters_seed = 42
    byzantine_seed = 83
    factor = 1.5

    train_amount = 6000
    test_amount = 1000

    x_train, y_train, x_test, y_test = dd.load_fashion_mnist_data()
    #dd.show_grid_fashion_mnist(x_train, y_train, 6, 6)

    ##train
    label_dict_train = dd.split_and_shuffle_labels(y_data=y_train, seed=seed, amount=train_amount)
    node_label_info_train, total_label_occurences_train, amount_info_table_train = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(
        number_of_samples=number_of_samples, n=n, amount=train_amount, seed=use_seed, min_n_each_node=min_n_each_node)

    x_train_dict, y_train_dict = dd.distribute_fashion_data_to_participants(label_dict=label_dict_train,
                                                                            amount=train_amount,
                                                                            number_of_samples=number_of_samples,
                                                                            n=n, x_data=x_train,
                                                                            y_data=y_train,
                                                                            node_label_info=node_label_info_train,
                                                                            amount_info_table=amount_info_table_train,
                                                                            x_name="x_train",
                                                                            y_name="y_train")

    ## test
    label_dict_test = dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount)
    node_label_info_test, total_label_occurences_test, amount_info_table_test = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(
        number_of_samples=number_of_samples,
        n=n, amount=test_amount, seed=use_seed, min_n_each_node=min_n_each_node)
    x_test_dict, y_test_dict = dd.distribute_fashion_data_to_participants(label_dict=label_dict_test,
                                                                          amount=test_amount,
                                                                          number_of_samples=number_of_samples,
                                                                          n=n, x_data=x_test,
                                                                          y_data=y_test,
                                                                          node_label_info=node_label_info_test,
                                                                          amount_info_table=amount_info_table_test,
                                                                          x_name="x_test",
                                                                          y_name="y_test")

    x_train = x_train.view(-1, 1, 28, 28)
    x_test = x_test.view(-1, 1, 28, 28)

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

    main_model = cm.Net_fashion()
    main_model = main_model.to(device)
    cm.weights_init(main_model)

    main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum,
                                     weight_decay=weight_decay)
    main_criterion = nn.CrossEntropyLoss()
    model_dict, optimizer_dict, criterion_dict = tn.create_model_optimizer_criterion_dict_for_fashion_mnist(
        number_of_samples, learning_rate,
        momentum, device, weight_decay)

    testAcc,testLoss=[],[]


    byzantine_node_list = dd.get_byzantine_node_list(hostile_node_percentage, number_of_samples, hostility_seed)
    np.random.seed(byzantine_seed)
    byzantine_seeds_array = np.random.choice(5000, size=iteration_num, replace=False)


    for iteration in range(iteration_num):
        print('iteration:',iteration+1)
        model_dict = tn.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict,
                                                                       number_of_samples)

        if is_organized:
            iteration_byzantine_seed = byzantine_seeds_array[iteration]
        else:
            iteration_byzantine_seed = None

        tn.start_train_end_node_process_byzantine(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                                  batch_size, model_dict, criterion_dict, optimizer_dict,
                                                  numEpoch, byzantine_node_list, byzantine_mean, byzantine_std,
                                                  device, iteration_byzantine_seed)


        print(byzantine_node_list)
        main_model = tn.bulyan(main_model,model_dict,number_of_samples,neighbors,device)

        test_loss, test_accuracy = tn.validation(main_model, test_dl, main_criterion, device)

        testAcc.append(test_accuracy)
        testLoss.append(test_loss)

        print("------------", ":accuracy: {:7.4f}".format(test_accuracy))
        print("------------", "-----loss: {:7.4f}".format(test_loss))

    return [testAcc, "bulyan", testLoss]
