class Instance(object):
    def __init__(self, vm_type: int, vm_duration: int, vm_start: int, vm_end: int, task_names: list):
        self.vm_type = vm_type
        self.vm_duration = vm_duration
        self.vm_start = vm_start
        self.vm_end = vm_end
        self.task_names = task_names
        self.properties = {}
