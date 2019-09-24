from network import Network
from utils import LOG_INFO
from layers import Linear, Relu, Sigmoid
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d


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


def train(model, loss, train_data, test_data, train_label, test_label):
    best_test_loss = -1
    update_round_before = 0
    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        train_loss_now, train_acc_now = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            test_loss_now, test_acc_now = test_net(model, loss, test_data, test_label, config['batch_size'])
            if best_test_loss == -1:
                update_round_before = 0
                best_test_loss = test_loss_now
            elif test_loss_now <= best_test_loss:
                update_round_before = 0
                best_test_loss = test_loss_now
            else:
                update_round_before += 1
                if update_round_before >= 5:
                    break


config = {
    "name": "linear_lr_0.1",
    "learning_rate": 0.1,
    "weight_decay": 0.0,
    "momentum": 0.0,
    "batch_size": 100,
    "max_epoch": 1000,
    "disp_freq": -1,
    "test_epoch": 1,
    "use_layer": [
        {
            "type": "Linear",
            "in_num": 784,
            "out_num": 10
        }
    ],
    "use_loss": "EuclideanLoss"
}

train_data, test_data, train_label, test_label = load_mnist_2d('data')
LOG_INFO('Using config %s now' % (config['name']))
model, loss = build_model(config)
train(model, loss, train_data, test_data, train_label, test_label)
