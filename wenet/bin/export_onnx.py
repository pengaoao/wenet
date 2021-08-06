# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import os

import torch
import onnx, onnxruntime
import yaml
import numpy as np

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint

def to_numpy(x):
    return x.detach().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    # parser.add_argument('--output_file', required=True, help='output file')
    # parser.add_argument('--output_onnx_file', required=True, help='output onnx file')
    # parser.add_argument('--output_quant_file',
    #                     default=None,
    #                     help='output quantized model file')
    args = parser.parse_args()
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_asr_model(configs)
    print(model)

    load_checkpoint(model, args.checkpoint)
    # Export jit torch script model

    model.eval()
    encoder = model.encoder
    encoder.set_onnx_mode(True)
    encoder.forward = encoder.forward_chunk

    batch_size = 1
    audio_len = 131
    x = torch.randn(batch_size, audio_len, 80, requires_grad=False)
    i4 = torch.tensor(0)
    i5 = torch.tensor(0)
    i1 = torch.randn(batch_size, 1, 256, requires_grad=False)
    i2 = torch.randn(12, batch_size, 1, 256, requires_grad=False)
    i3 = torch.randn(12, batch_size, 256, 7, requires_grad=False)

    onnx_path = 'encoder.onnx'
    torch.onnx.export(encoder,
                    (x, i4, i5, i1, i2, i3),
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input', 'i4', 'i5', 'i1', 'i2', 'i3'],
                    output_names=['output', 'o1', 'o2', 'o3'],
                    dynamic_axes={'input': [1], 'i1':[1], 'i2':[2],
                                    'output': [1], 'o1':[1], 'o2':[2]},
                    verbose=True
                    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("encoder onnx_model check pass!")

    # compare ONNX Runtime and PyTorch results
    encoder.set_onnx_mode(False)
    y, o1, o2, o3 = encoder(x, i4, i5, None, None, i3)

    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x),
                ort_session.get_inputs()[1].name: to_numpy(i4),
                ort_session.get_inputs()[2].name: to_numpy(i5),
                ort_session.get_inputs()[3].name: to_numpy(i1),
                ort_session.get_inputs()[4].name: to_numpy(i2),
                ort_session.get_inputs()[5].name: to_numpy(i3),
                }
    ort_outs = ort_session.run(None, ort_inputs)

    # np.testing.assert_allclose(to_numpy(y), ort_outs[0][:, 1:, :], rtol=1e-05, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(o1), ort_outs[1][:, 1:, :], rtol=1e-05, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(o2), ort_outs[2][:, :, 1:, :], rtol=1e-05, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(o3), ort_outs[3], rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(to_numpy(y), ort_outs[0][:, :, :], rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(to_numpy(o1), ort_outs[1][:, :, :], rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(to_numpy(o2), ort_outs[2][:, :, :], rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(to_numpy(o3), ort_outs[3], rtol=1e-05, atol=1e-05)
    print("Exported encoder model has been tested with ONNXRuntime, and the result looks good!")

    #export decoder onnx
    decoder = model.decoder
    onnx_path = 'decoder.onnx'
    memory = torch.randn(10, 131, 256)
    memory_mask = torch.ones(10, 1, 131).bool()
    ys_in_pad = torch.randint(0, 4232, (10, 50)).long()
    ys_in_lens = torch.tensor([13, 13, 13, 13, 13, 13, 13, 13, 50, 13])
    r_ys_in_pad = torch.randint(0, 4232, (10, 50)).long()

    torch.onnx.export(decoder,
                    (memory, memory_mask, ys_in_pad, ys_in_lens, r_ys_in_pad),
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['memory', 'memory_mask', 'ys_in_pad', 'ys_in_lens', 'r_ys_in_pad'],
                    output_names=['l_x', 'r_x', 'olens'],
                    dynamic_axes={'memory': [1], 'memory_mask':[2], 'ys_in_pad':[2],
                                    'ys_in_lens': [0], 'r_ys_in_pad':[1]},
                    verbose=True
                    )
