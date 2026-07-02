import json
from pathlib import Path


MODEL_PATH = {
    # plms
    'qwen3-1.7b' : '/home/test/testdata/luoyuqi/plms/Qwen3-1.7B',
    'qwen3-8b' : '/home/test/testdata/luoyuqi/datasets/cache/models/Qwen/Qwen3-8B',
    'nemotron-1.5b' : '/home/test/testdata/luoyuqi/plms/OpenMath-Nemotron-1.5B',
    'distil-1.5b' : '/home/test/testdata/luoyuqi/plms/DeepSeek-R1-Distill-Qwen-1.5B',
}

checkpoints_config_path = Path(__file__).with_name('checkpoints_config.json')
if checkpoints_config_path.exists():
    with checkpoints_config_path.open() as f:
        checkpoint_configs = json.load(f)

    config_stack = [checkpoint_configs]
    while config_stack:
        current_config = config_stack.pop()
        for key, value in current_config.items():
            if isinstance(value, str):
                MODEL_PATH[key] = value
            else:
                config_stack.append(value)

PLMS = ['qwen3-1.7b', 'qwen3-8b', 'nemotron-1.5b', 'distil-1.5b']

def get_tokenizer_path(model_name: str):
    for key in PLMS:
        if key in model_name:
            return MODEL_PATH[key]
    raise ValueError(f'Tokenizer path for model {model_name} not found.')
