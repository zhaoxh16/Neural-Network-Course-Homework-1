from network import Network
from utils import LOG_INFO
from layers import Linear, Relu, Sigmoid
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import json
import os
import matplotlib.pyplot as plt
import matplotlib
import copy
matplotlib.use('AGG')


def build_model(config):
    model = Network()
    layer_num = 0
    for layer in config['use_layer']:
        if layer['type'] == "Linear":
            in_num = layer['in_num']
            out_num = layer['out_num']
            if "init_std" in layer.keys():
                model.add(Linear(layer['type']+str(layer_num), in_num, out_num, init_std=layer['init_std']))
            else:
                model.add(Linear(layer['type']+str(layer_num), in_num, out_num))
            layer_num += 1
        elif layer['type'] == 'Relu':
            model.add(Relu(layer['type'] + str(layer_num)))
            layer_num += 1
        else:
            assert 0
    loss_name = config['use_loss']
    if loss_name == 'EuclideanLoss':
        loss = EuclideanLoss(loss_name)
    elif loss_name == 'SoftmaxCrossEntropyLoss':
        loss = SoftmaxCrossEntropyLoss(loss_name)
    else:
        assert 0
    return model, loss


def train_and_save(model, loss, train_data, test_data, train_label, test_label):
    best_test_loss = -1
    update_round_before = 0
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    epoch_final = 0
    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        train_loss_now, train_acc_now = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        train_loss_list.append(train_loss_now)
        train_acc_list.append(train_acc_now)
        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_loss_now, test_acc_now = test_net(model, loss, test_data, test_label, config['batch_size'])
            test_loss_list.append(test_loss_now)
            test_acc_list.append(test_acc_now)
            if best_test_loss == -1:
                update_round_before = 0
                best_test_loss = test_loss_now
            elif test_loss_now <= best_test_loss:
                update_round_before = 0
                best_test_loss = test_loss_now
            else:
                update_round_before += 1
                if update_round_before >= 5:
                    epoch_final = epoch + 1
                    break
    save_dir = os.path.join('result', config['name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_dict = {
        "train_loss": train_loss_list,
        "train_acc": train_acc_list,
        "test_loss": test_loss_list,
        "test_acc": test_acc_list
    }
    with open(os.path.join(save_dir, "result.json"), 'w') as f:
        json.dump(result_dict, f)
    x = range(epoch_final)
    plt.cla()
    plt.plot(x, train_loss_list, label="train loss")
    plt.plot(x, test_loss_list, label="test loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.cla()
    plt.plot(x, train_acc_list, label="train acc")
    plt.plot(x, test_acc_list, label="test acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "acc.png"))


train_data, test_data, train_label, test_label = load_mnist_2d('data')

with open("config.json", 'r') as f:
    config_all = json.load(f)

while len(config_all['train_config']) != 0:
    train_config = config_all['train_config'][0]
    config = copy.copy(config_all['default_config'])
    for key, value in train_config.items():
        config[key] = value
    LOG_INFO('Using config %s now' % (config['name']))
    model, loss = build_model(config)
    train_and_save(model, loss, train_data, test_data, train_label, test_label)
    config_all['finish_config'].append(train_config)
    del config_all['train_config'][0]
    with open("config.json", 'w') as f:
        json.dump(config_all, f)
