import matplotlib
import matplotlib.pyplot as plt
import json
import os
matplotlib.use('AGG')


experiment_names = [
    # "linear_lr_0.01",
    # "linear_lr_0.001",
    # "linear_lr_0.0001"

    # "linear_lr_0.001",
    # "linear_wd_0.001",
    # "linear_wd_0.01",
    # "linear_wd_0.1"

    # "linear_lr_0.001",
    # "linear_mm_0.001",
    # "linear_mm_0.01",
    # "linear_mm_0.1"

    # "linear_lr_0.001",
    # "linear_softmax"

    # "linear_softmax",
    # "linear_relu_softmax",

    # "linear_relu_linear_euclidean",
    # "linear_relu_linear_softmax",
    # "linear_sigmoid_linear_euclidean",
    # "linear_sigmoid_linear_softmax"

    # "linear_relu_linear_euclidean",
    # "linear_relu_linear_softmax",
    # "linear_relu_linear_relu_linear_euclidean",
    # "linear_relu_linear_relu_linear_softmax",

    # "linear_sigmoid_linear_sigmoid_linear_euclidean",
    # "linear_sigmoid_linear_sigmoid_linear_softmax",
    # "linear_relu_linear_relu_linear_euclidean",
    # "linear_relu_linear_relu_linear_softmax"

    "linear_sigmoid_linear_euclidean",
    "linear_sigmoid_linear_softmax",
    "linear_sigmoid_linear_sigmoid_linear_euclidean",
    "linear_sigmoid_linear_sigmoid_linear_softmax"
]

result_dict = {}
for name in experiment_names:
    result_path = os.path.join(os.path.join("result", name), "result.json")
    with open(result_path, 'r') as f:
        result = json.load(f)
        result_dict[name] = result

plt.cla()
for name, result in result_dict.items():
    plt.plot(range(len(result["test_acc"])), result["test_acc"], label=name)
plt.legend()
plt.savefig(os.path.join("analyze", "experiment_2.png"))
