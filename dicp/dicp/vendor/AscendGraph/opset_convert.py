import torch
import torch_dipu
from dicp.dynamo_bridge.compile_fx import is_torch_210
from dicp.vendor.AscendGraph.ascend_op import MatMul, CastToCpu, IdentityInp, InplaceCopyWithOffset
from dicp.vendor.AscendGraph.conversion import AtenToAscendTransformer

if is_torch_210:
    from dicp.dynamo_bridge.op_transformer import BackendPatternMatcherTransformer
    from dicp.vendor.AscendGraph.pattern_replacement import (
        ascend_pattern_matcher,
        aten_patterns_cls_list,
        ascend_patterns_cls_list
    )


class ArgsTransDataPass:
    def transform(self, gm: torch.fx.graph_module):
        for n in gm.graph.nodes:
            if hasattr(n, 'op') and n.op == 'placeholder':
                fake_tensor = n.meta['val']
                memo = fake_tensor.fake_mode.fake_tensor_converter.tensor_memo     
                for key in memo:
                    if id(memo[key].fake_device) == id(fake_tensor.fake_device):
                        memory_format = torch_dipu.get_native_memory_format(key())
                        n.meta['native_memory_format'] = str(memory_format.name)
                        break
        return gm


class OutputMarkPass:
    def __init__(self):
        self.assign_args = []
        self.cpu_tensor = []
        self.assign_with_offset_args = {}

    def transform(self, gm: torch.fx.graph_module):
        # dynamic shape feature
        input_names = []
        for n in gm.graph.nodes:
            if n.op == 'placeholder':
                input_names.append(n.name)

        for n in gm.graph.nodes:
            if n.op != 'call_function':
                continue
            if type(n.target) == CastToCpu:
                self.cpu_tensor.append(n.name)
            elif type(n.target) == InplaceCopyWithOffset:
                input_index = input_names.index(str(n.args[0]))
                offset = int(n.args[-1])
                self.assign_with_offset_args[n.name] = {'name': n.name, 'input_index': input_index, 'offset': offset}
            elif type(n.target) == IdentityInp:
                if len(n.args) == 2 and n.args[1] is not None and str(n.args[1]) in input_names:
                    self.assign_args.append((n.name, input_names.index(str(n.args[1]))))
                else:
                    raise RuntimeError("Op inner copy_ error!")

        for n in gm.graph.nodes:
            if n.op == 'call_function':
                prop = {}
                if n.name in self.cpu_tensor:
                    prop.update({'cpu_tensor' : n.name})
                if len(self.assign_args) > 0 and n.name in list(zip(*self.assign_args))[0]:
                    idx = list(zip(*self.assign_args))[0].index(n.name)
                    prop.update({'assign_args' : (self.assign_args[idx][0], self.assign_args[idx][1])})
                if n.name in self.assign_with_offset_args.keys():
                    prop['assign_with_offset_args'] = self.assign_with_offset_args[n.name]
                n.meta['prop'] = prop
        return gm


def symint_in_inputs(nodes):
    # dynamic shape feature
    for node in nodes:
        if node.op == 'placeholder':
            if hasattr(node, 'meta'):
                node = node.meta['val']
            if isinstance(node, torch.SymInt):
                return True
            if hasattr(node, 'shape'):
                for dim in node.shape:
                    if isinstance(dim, torch.SymInt):
                        return True
    return False

def ascendgraph_opset_convert(
    gm: torch.fx.GraphModule,
):
    if is_torch_210:
        gm = BackendPatternMatcherTransformer(
            ascend_pattern_matcher, aten_patterns_cls_list).transform(gm)
    gm = AtenToAscendTransformer(gm).transform()

    # For bug in pytorch
    # Avoid for dynamic shape
    if is_torch_210 and not symint_in_inputs(list(gm.graph.nodes)):
        gm = BackendPatternMatcherTransformer(
            ascend_pattern_matcher, ascend_patterns_cls_list).transform(gm)
    gm = OutputMarkPass().transform(gm)
    # uncomment this after DIOPI support pytorch2.1.1
    # gm = ArgsTransDataPass().transform(gm)
    return gm
