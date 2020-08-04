class NewInstance(object):
    def __init__(self, vm_type: int, vm_cost: int, vm_start: int, vm_end: int, pcp: list):
        self.vm_type = vm_type
        self.vm_cost = vm_cost
        self.vm_start = vm_start
        self.vm_end = vm_end
        self.task_list = pcp
        self.properties = {}
        self.task_names = []

    def get_duration(self):
        return self.vm_end - self.vm_start

    def get_cost(self):
        return (self.vm_end - self.vm_start) * self.vm_cost
