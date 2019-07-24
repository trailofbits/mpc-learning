from queue import Queue

class Gate():
    def __init__(self, id_num, gate_type, input_ids, gate_queue=None, ready=False, const_input=None):
        self.id_num = id_num
        self.gate_type = gate_type
        self.input_ids = input_ids
        self.inputs = {}
        for in_id in input_ids:
            self.inputs[in_id] = ""
        if self.gate_type == "INPUT":
            self.inputs = {}
            self.inputs[""] = ""
        #self.gate_queue = gate_queue
        self.ready = ready
        self.complete = False
        self.output = ""
        self.const_input = const_input
    
    def reset(self):
        for in_id in self.inputs:
            self.inputs[in_id] = ""
        if self.gate_type == "INPUT":
            self.inputs[""] = ""
        self.ready = False

    def add_input(self, in_id, in_value):
        #if in_id == "":
        #    if type(in_value) == list:
        #        print("in_val: " + str(in_value[0].get_x()) + ", " + str(in_value[0].get_a()))
        self.inputs[in_id] = in_value
        #if self._is_ready():
        #    print("we ready")
        #    try:
        #        self.gate_queue.put(self)
        #    except:
        #        print("no gate queue for gate: " + self.id_num)

    def get_inputs(self):
        input_vals = []
        for key in self.inputs:
            if key != self.const_input:
                input_vals.append(self.inputs[key])
        return input_vals

    def get_input_ids(self):
        return self.input_ids

    def get_type(self):
        return self.gate_type
    
    def get_id(self):
        return self.id_num

    def get_const_inputs(self):
        input_vals = []
        for key in self.inputs:
            if key == self.const_input:
                input_vals.append(self.inputs[key])
        return input_vals

    def is_ready(self):
        ready = True
        for in_id in self.inputs:
            if self.inputs[in_id] == "":
                ready = False
        self.ready = ready
        return self.ready

    def is_complete(self):
        return self.complete

    def set_queue(self,q):
        self.gate_queue = q

    
    
