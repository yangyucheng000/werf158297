# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Functions of cells"""
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore.ops import functional as F

from src.args import args


class OutputTo16(nn.Cell):
    "Wrap cell for amp. Cast network output back to float16"

    def __init__(self, op):
        super(OutputTo16, self).__init__(auto_prefix=False)
        self._op = op

    def construct(self, x):
        return F.cast(self._op(x), mstype.float16)


def do_keep_fp16(network, cell_types):
    """Cast cell to fp32 if cell in cell_types"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, cell_types):
            cell.to_float(mstype.float16)


def do_keep_fp32(network, cell_types):
    """Cast cell to fp32 if cell in cell_types"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, cell_types):
            cell.to_float(mstype.float32)


def cast_amp(net):
    """cast network amp_level"""
    if args.amp_level == "O2":
        cell_types = (nn.LayerNorm, nn.BatchNorm2d)
        print(f"=> using amp_level {args.amp_level}")
        net.to_float(mstype.float16)
        do_keep_fp32(net, cell_types)
    elif args.amp_level == "O1":
        cell_types = (nn.LayerNorm, nn.Softmax, nn.BatchNorm2d, nn.GELU, nn.SyncBatchNorm, nn.GroupNorm)
        print(f"=> using amp_level {args.amp_level}")
        net.to_float(mstype.float16)
        do_keep_fp32(net, cell_types)
    elif args.amp_level == "O3":
        print(f"=> using amp_level {args.amp_level}")
        net.to_float(mstype.float16)
    else:
        print(f"=> using amp_level {args.amp_level}")
        args.loss_scale = 1.
        args.is_dynamic_loss_scale = 0
        print(f"=> When amp_level is O0, using fixed loss_scale with {args.loss_scale}")


class WithEvalCell(nn.Cell):
    def __init__(self, network, loss_fn, add_cast_fp32=False):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self._network = network
        self._loss_fn = loss_fn
        self.add_cast_fp32 = validator.check_value_type("add_cast_fp32", add_cast_fp32, [bool], self.cls_name)

    def construct(self, data, label):
        outputs= self._network(data)
        #outputs = output1 + output2
        if self.add_cast_fp32:
            label = F.mixed_precision_cast(mstype.float32, label)
            outputs = F.cast(outputs, mstype.float32)
        loss = self._loss_fn(outputs, label)
        return loss, outputs, label


class InferenceCell(nn.Cell):
    def __init__(self, network):
        super(InferenceCell, self).__init__()
        self._network = network

    def construct(self, data):
        output = self._network(data)
        return output