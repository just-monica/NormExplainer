
import yaml

def gen_model_config(layers: int, types: str, hidden_size: int, model_path: str, norm_type: str):
    config =  {
        'layers': layers,
        'types': types,
        'hidden_size': hidden_size,
        'saved_path': model_path + '.pth',
        'Norm': norm_type,
    }
    with open(f'./Configs/ModelConfigs/{model_path}.yaml', 'w') as f:
        yaml.dump(config, f)
    print(f'Generated model config file at ./Configs/ModelConfigs/{model_path}.yaml')

def process_task_config(config):
    if config['norm_p'] == 'inf':
        config['norm_p'] = float('inf')
    elif config['norm_p'] == '-inf':
        config['norm_p'] = float('-inf')
    return config