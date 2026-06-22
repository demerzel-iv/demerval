MODEL_PATH = {
    # plms
    'qwen3-1.7b' : '/home/test/testdata/luoyuqi/plms/Qwen3-1.7B',
    'qwen3-8b' : '/home/test/testdata/luoyuqi/datasets/cache/models/Qwen/Qwen3-8B',
    'nemotron-1.5b' : '/home/test/testdata/luoyuqi/plms/OpenMath-Nemotron-1.5B',
    'distil-1.5b' : '/home/test/testdata/luoyuqi/plms/DeepSeek-R1-Distill-Qwen-1.5B',
}

TOKENIZER_PATH_DICT = {
    'qwen3-1.7b' : '/home/test/testdata/luoyuqi/plms/Qwen3-1.7B',
    'qwen3-8b' : '/home/test/testdata/luoyuqi/datasets/cache/models/Qwen/Qwen3-8B',
    'nemotron-1.5b' : '/home/test/testdata/luoyuqi/plms/OpenMath-Nemotron-1.5B',
    'distil-1.5b' : '/home/test/testdata/luoyuqi/plms/DeepSeek-R1-Distill-Qwen-1.5B',
}

def get_tokenizer_path(model_name: str):
    for key in TOKENIZER_PATH_DICT:
        if key in model_name:
            return TOKENIZER_PATH_DICT[key]
    raise ValueError(f'Tokenizer path for model {model_name} not found.')
