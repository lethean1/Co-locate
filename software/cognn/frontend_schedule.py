import threading
import torch
import importlib
import time

from util.util import timestamp
from cognn.policy import *
from cognn.schedule import Schedule

class FrontendScheduleThd(threading.Thread):
    def __init__(self, model_list, qin, worker_list):
        super(FrontendScheduleThd, self).__init__()
        self.model_list = model_list
        self.qin = qin
        self.worker_list = worker_list
        self.cur_w_idx = 0
        

    def run(self):
        timestamp('schedule', 'start')
        
        job_list = []
        while True:
            # Get request           
            agent, task_name, data_name, num_layers = self.qin.get()
            job_list.append([agent, task_name, data_name, num_layers])
            timestamp('schedule', 'get_request')
            if len(job_list) == len(self.model_list):
                break
        
        # run a job on the specified worker
        def run_job(job, w_idx):
            agent, task_name, data_name, num_layers = job[0], job[1], job[2], job[3]
            new_pipe, _, = self.worker_list[w_idx]

            # send request to new worker
            model_name = []
            for model in self.model_list:
                if model[0] == task_name and model[1] == data_name and model[2] == num_layers:
                    model_name = model

            new_pipe.send((agent, model_name))
            timestamp('schedule', 'notify_new_worker')

        # run the first job in the queue
        first_job = job_list.pop(0)
        timestamp('schedule', 'run first job')
        run_job(first_job, self.cur_w_idx)
        self.cur_w_idx += 1
        
        # get the co-locate model
        model_schedule = Schedule(job_list) 
        co_job, job_list = model_schedule.get_co_job()
        timestamp('schedule', 'run first co-job')
        run_job(co_job, self.cur_w_idx)
        self.cur_w_idx += 1
        
        # monitor jobs, schedule in time once a job is finished
        w_idx = 0
        while(len(job_list)!=0):
            while True:
                w_idx %= len(self.worker_list)
                new_pipe, _ = self.worker_list[w_idx]
                # Recv response
                if new_pipe.poll():
                    res = new_pipe.recv()
                    timestamp('schedule', 'a job is finished')
                    new_co_job, job_list = model_schedule.get_co_job()
                    timestamp('schedule', 'run a new co-job')
                    run_job(new_co_job, w_idx)
                    break
                w_idx += 1