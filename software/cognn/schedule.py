import torch

### Class
class Schedule():
    def __init__(self, job_list):
        """ """
        self.job_list = job_list

    def get_co_job(self):
        job = self.job_list.pop(0)
        return job, self.job_list