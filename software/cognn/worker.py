from queue import Queue
from multiprocessing import Process
import torch
import time

from cognn.worker_common import ModelSummary
from util.util import timestamp

class WorkerProc(Process):
    def __init__(self, model_list, pipe):
        super(WorkerProc, self).__init__()
        self.model_list = model_list
        self.pipe = pipe
        
    def run(self):
        timestamp('worker', 'start')

        # Warm up CUDA 
        torch.randn(1024, device='cuda')
        time.sleep(1)
        while True:  # dispatch workers for task execution
            agent, model_name = self.pipe.recv()
            model_summary = ModelSummary(model_name)
            timestamp('worker', 'import models')
            timestamp('worker', 'after importing models')

            # start doing training
            with torch.cuda.stream(model_summary.cuda_stream_for_computation):
                output = model_summary.execute()
                print('output-{}: {}'.format(model_name, output))
                timestamp('worker', 'one job finished')
#                print ('Training time: {} ms'.format(output))
                del output

                self.pipe.send('FNSH')
                agent.send(b'FNSH')

            timestamp('worker_comp_thd', 'complete')
