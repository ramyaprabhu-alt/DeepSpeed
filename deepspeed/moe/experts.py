# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import copy


class Experts(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        super(Experts, self).__init__()

        self.deepspeed_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.deepspeed_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs):
        # print("x shape: {}".format(inputs.shape))
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # l = []
        for chunk, expert in zip(chunks, self.deepspeed_experts):
            # print("forward pass on expert")
            # print("chunk shape: {}".format(chunk.shape))
            # start.record()
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