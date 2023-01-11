import json
import time

def log_param(param_name, param_value, episode, file_name):
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
    except:
        data = {}
    if param_name in data:
        data[param_name]['values'].append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
            'value': param_value,
            'episode': episode
        })
    else:
        data[param_name] = {
            'values': [{
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                'value': param_value,
                'episode': episode
            }]
        }
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=1)