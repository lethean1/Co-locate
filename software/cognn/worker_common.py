import importlib

import torch

### Class
class ModelSummary():
    def __init__(self, model_name):
        """ """
        self.task_name, self.data_name, self.num_layers = model_name[0], model_name[1], int(model_name[2])
        self.load_model()

    def execute(self): 
        return self.func(self.model, self.data)

    def load_model(self):
        model_module = importlib.import_module('task.' + self.task_name)
        self.model, self.func, _ = model_module.import_task(self.data_name, self.num_layers)
        _, self.data = model_module.import_model(self.data_name, self.num_layers)
        
        self.cuda_stream_for_computation = torch.cuda.Stream()
        
        print('{} {} {}'.format(self.task_name, self.data_name, self.num_layers))   
