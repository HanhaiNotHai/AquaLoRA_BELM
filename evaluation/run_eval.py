import sys

sys.path.append('..')

import os
import types

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear

from utils.lora_modules import (
    CustomLoRACompatibleConvforward,
    CustomLoRACompatibleLinearforward,
    CustomLoRAConv2dLayerforward,
    CustomLoRALinearLayerforward,
)
from utils.models import MapperNet, SecretDecoder


@torch.inference_mode()
def main():

    device = torch.device('cuda')
    bs = 8
    guidance_scale = 7.5

    rank = 320
    msg_bits = 48
    pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'
    model_path = '../AquaLoRA-Models/ppft_trained'
    lora_weight_name = 'pytorch_lora_weights.safetensors'
    mapper_weight_name = 'mapper.pt'
    msgdecoder_weight_name = 'msgdecoder.pt'
    prompt_path = 'prompt.txt'

    pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
    pipeline.load_lora_weights(model_path, weight_name=lora_weight_name)
    pipeline.to(device)

    for name, module in pipeline.unet.named_modules():
        if isinstance(module, LoRACompatibleConv):
            module.forward = types.MethodType(CustomLoRACompatibleConvforward, module)
            if module.lora_layer is not None:
                module.lora_layer.forward = types.MethodType(
                    CustomLoRAConv2dLayerforward, module.lora_layer
                )
        elif isinstance(module, LoRACompatibleLinear):
            module.forward = types.MethodType(CustomLoRACompatibleLinearforward, module)
            if module.lora_layer is not None:
                module.lora_layer.forward = types.MethodType(
                    CustomLoRALinearLayerforward, module.lora_layer
                )

    mapper = MapperNet(input_size=msg_bits, output_size=rank)
    mapper.load_state_dict(torch.load(os.path.join(model_path, mapper_weight_name)))
    mapper.eval()
    mapper.to(device)

    msgdecoder = SecretDecoder(output_size=msg_bits)
    msgdecoder.load_state_dict(torch.load(os.path.join(model_path, msgdecoder_weight_name)))
    msgdecoder.eval()
    msgdecoder.to(device)

    with open(prompt_path, 'r') as f:
        prompts = f.readlines()
    prompts = [prompt.strip() for prompt in prompts]
    prompts = prompts[:bs]

    msg = torch.randint(0, 2, (bs, msg_bits)).to(device)
    msg_ = msg.float()
    mapped_loradiag = mapper(msg_)

    if guidance_scale > 1:
        mapped_loradiag = torch.cat([mapped_loradiag] * 2)

    images = pipeline(
        prompts,
        guidance_scale=guidance_scale,
        cross_attention_kwargs={'scale': mapped_loradiag},
    ).images

    # decode messages
    np_images = np.stack([np.asarray(img) for img in images])
    tensor_images = torch.from_numpy(np_images).to(device)

    tensor_images = tensor_images.permute(0, 3, 1, 2)
    tensor_images = tensor_images / 127.5 - 1

    decoded_msg = msgdecoder(tensor_images.float())
    # calulate accuracy
    decoded_msg = torch.argmax(decoded_msg, dim=-1)
    res = msg - decoded_msg
    valid_accuracy = (res == 0).float().mean()
    print(f"validation accuracy: {valid_accuracy.item()}")


if __name__ == "__main__":
    main()
