from tdc.benchmark_group import admet_group

group = admet_group(path = 'input_data/tdcommons/')
benchmark = group.get('ames')

name = benchmark['name']
train_val, test = benchmark['train_val'], benchmark['test']