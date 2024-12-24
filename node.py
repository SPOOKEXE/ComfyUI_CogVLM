
import os
import sys

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from transformers import AutoModelForCausalLM, LlamaTokenizer
from PIL import Image

import torch

def caption_cogvlm(img, max_length):
    tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to('cuda').eval()

    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
        'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_length": max_length, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(outputs[0])

class CogVLMCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image" : ('IMAGE', {}),
                "max_length" : ('INT', {})
            }
        }
    CATEGORY = "caption"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", )

    def main(self, image, max_length):
        result = caption(image, max_length)
        return (result, )