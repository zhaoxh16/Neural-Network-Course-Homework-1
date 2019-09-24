import pandas as pd
from pandas import Series, DataFrame
import json
import os


ignore_keys = ['use_layer', 'max_epoch', 'disp_freq', 'test_epoch']

with open("config.json", 'r') as f:
    config = json.load(f)
    default_config = config['default_config']
    config_item_num = len(default_config.keys())-1
    finish_configs = config['finish_config']
    experiment_count = len(finish_configs)
    csv_data = {}
    for config_detail in finish_configs:
        config_name = config_detail['name']
        with open(os.path.join(os.path.join("result", config_name), "result.json"), 'r') as result_file:
            result = json.load(result_file)
            config_detail['train_time'] = result['train_time']
            config_detail['train_epochs'] = len(result['train_loss'])
            config_detail['min_loss'] = min(result['test_loss'][-6:])
            config_detail['max_acc'] = max(result['test_acc'][-6:])
        for key, value in default_config.items():
            if key in ignore_keys:
                continue
            if key not in config_detail:
                config_detail[key] = value
        for key, value in config_detail.items():
            if key in ignore_keys:
                continue
            if key not in csv_data.keys():
                csv_data[key] = []
            csv_data[key].append(config_detail[key])
    df = DataFrame(csv_data)
    df.to_csv("experiment_data.csv")
