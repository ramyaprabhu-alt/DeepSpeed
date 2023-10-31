# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import copy
import copyreg

class Experts(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        super(Experts, self).__init__()
        print("in experts.py __init__")
        print("expert type: {}".format(type(expert)))
        try:
            self.deepspeed_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
        except:
            copyreg.pickle(expert.__class__, expert.pickle_myself, expert.unpickle_myself)
            self.deepspeed_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
      

        # for i in self.deepspeed_experts:
            # print("expert device: {}".format(i))
        # exit(0)
        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.deepspeed_experts:
            
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                # print("param name: {}".format(name))
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs):
        # print("x shape: {}".format(inputs.shape))
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # l = []
        count = 0
        for chunk, expert in zip(chunks, self.deepspeed_experts):
            # print("forward pass on expert")
            # print("chunk shape: {}".format(chunk.shape))
            # start.record()
            count+= 1
            # print("forward pass on expert {}".format(count))
            print("GPU: {}".format(torch.cuda.current_device()))
            # print("GPU check : {}".format(expert.device()))
            print("chunk size", chunk.shape)
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]
            # end.record()
            # torch.cuda.synchronize()
            # print("forward pass on experts took: {} ms".format(start.elapsed_time(end)))
            # l.append(start.elapsed_time(end))
        # print("forward pass per experts took: {} ms on avg".format(sum(l)/len(l)))
        expert_output = torch.cat(expert_outputs, dim=1)
        
        return expert_output
