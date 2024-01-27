import torch

# from moellava.constants import X_TOKEN_INDEX
from transformers import TextStreamer

from moellava.constants import IMAGE_TOKEN_INDEX
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria, tokenizer_image_token
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init

# <a href="https://arxiv.org/pdf/2310.01852.pdf"><img src="https://img.shields.io/badge/Arxiv-2310.01852-red"></a>
title_markdown = ("""
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <a href="https://github.com/PKU-YuanGroup/MoE-LLaVA" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
    <img src="https://s11.ax1x.com/2023/12/28/piqvDMV.png" alt="MoE-LLaVA🚀" style="max-width: 120px; height: auto;">
  </a>
  <div>
    <h1 >MoE-LLaVA: Mixture of Experts for Large Vision-Language Models</h1>
    <h5 style="margin: 0;">If you like our project, please give us a star ✨ on Github for the latest update.</h5>
  </div>
</div>


<div align="center">
    <div style="display:flex; gap: 0.25rem;" align="center">
        <a href='https://github.com/PKU-YuanGroup/MoE-LLaVA'><img src='https://img.shields.io/badge/Github-Code-blue'></a>
        <a href='https://github.com/PKU-YuanGroup/MoE-LLaVA/stargazers'><img src='https://img.shields.io/github/stars/PKU-YuanGroup/MoE-LLaVA.svg?style=social'></a>
    </div>
</div>
""")

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")

learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")


class Chat:
    def __init__(self, model_path, conv_mode, model_base=None, load_8bit=False, load_4bit=False, device='cuda'):
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                                   load_8bit, load_4bit,
                                                                                   device=device)
        self.image_processor = processor['image']
        self.video_processor = processor['video']
        self.conv_mode = conv_mode
        self.device = self.model.device
        print(self.model)

    def get_prompt(self, qs, state):
        state.append_message(state.roles[0], qs)
        state.append_message(state.roles[1], None)
        return state

    @torch.inference_mode()
    def generate(self, images_tensor: list, prompt: str, first_run: bool, state):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        state = self.get_prompt(prompt, state)
        prompt = state.get_prompt()
        print('\n\n\n')
        print(prompt)

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            self.device)

        temperature = 0.2

        max_new_tokens = 1024

        stop_str = conv_templates[self.conv_mode].copy().sep if conv_templates[
                                                                    self.conv_mode].copy().sep_style != SeparatorStyle.TWO else \
        conv_templates[self.conv_mode].copy().sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        print('response', outputs)
        return outputs, state

