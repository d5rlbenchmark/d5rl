import os

DATA_PATH = '/home/asap7772/binsort_bridge_sorted/04_26_collect_multitask/'
ALL_DATASETS = [
    'actionnoise0.0_binnoise0.0_policypickplace_sparse0', 
    'actionnoise0.05_binnoise0.1_policypickplace_sparse0', 
    'actionnoise0.0_binnoise0.0_policysorting_sparse0', 
    'actionnoise0.05_binnoise0.0_policysorting_sparse0', 
    'actionnoise0.0_binnoise0.1_policypickplace_sparse0', 
    'actionnoise0.05_binnoise0.1_policysorting_sparse0', 
    'actionnoise0.05_binnoise0.0_policypickplace_sparse0', 
    'actionnoise0.0_binnoise0.1_policysorting_sparse0'
]

def debug_config():
    return ["/home/asap7772/binsort_bridge/test/actionnoise0.0_binnoise0.0_policysorting_sparse0/train/out.npy"]

def sorting_dataset(train=True):
    suffix = 'train' if train else 'test'
    suffix = "/" + suffix + "/" + 'out.npy'
    return [DATA_PATH + x + suffix for x in ALL_DATASETS if 'sorting' in x]

def pickplace_dataset(train=True):
    suffix = 'train' if train else 'test'
    suffix = "/" + suffix + "/" + 'out.npy'
    return [DATA_PATH + x + suffix for x in ALL_DATASETS if 'pickplace' in x]

def sorting_pickplace_dataset(train=True):
    suffix = 'train' if train else 'test'
    suffix = "/" + suffix + "/" + 'out.npy'
    return [DATA_PATH + x + suffix for x in ALL_DATASETS]

if __name__ == '__main__':
    print(sorting_dataset())
    print(sorting_pickplace_dataset())
    print(pickplace_dataset())
    
    for y in [sorting_dataset(), sorting_pickplace_dataset(), pickplace_dataset()]:
        for x in y:
            assert os.path.exists(x)