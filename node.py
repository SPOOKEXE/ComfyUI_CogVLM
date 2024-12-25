
import os
import sys

sys.path.append(
	os.path.dirname(os.path.abspath(__file__))
)

from transformers import AutoModelForCausalLM, LlamaTokenizer
from PIL import Image
from torchvision import transforms
# from bitsandbytes import Int4Quantizer

import torchvision.transforms as T
import torch
import numpy as np

def caption_cogvlm(image, max_new_tokens):
	print("Captioning image with CogVLM")
	tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
	model = AutoModelForCausalLM.from_pretrained(
		'THUDM/cogvlm-chat-hf',
		torch_dtype=torch.bfloat16,
		low_cpu_mem_usage=True,
		trust_remote_code=True
	).to('cuda').eval()

	# quantizer = Int4Quantizer(model)
	# model = quantizer.quantize()
	# model = model.eval().to('cuda')

	# Process the image
	image = image.squeeze(0).permute(2, 0, 1)
	image = T.ToPILImage()(image)

	print(image)

	query = 'Describe the content of the image in a clear and concise sentence, focusing on the main objects, actions, and context, while maintaining natural language fluency'
	inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])
	inputs = {
		'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
		'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
		'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
		'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
	}
	gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}

	with torch.no_grad():
		outputs = model.generate(**inputs, **gen_kwargs)
		outputs = outputs[:, inputs['input_ids'].shape[1]:]
	return outputs

class CogVLMCaption:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image" : ('IMAGE', {}),
				"max_new_tokens" : ('INT', {})
			}
		}
	CATEGORY = "caption"
	FUNCTION = "main"
	RETURN_TYPES = ("STRING", )

	def main(self, image, max_new_tokens):
		result = caption_cogvlm(image, max_new_tokens)
		return (result, )
