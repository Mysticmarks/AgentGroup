'''
Code copied from Generative Agents
'''
import time

import os

import copy
from zhipuai import ZhipuAI

from config import *
from .utils import *
from .hunyuan import HunYuan_request


class ModelPool:
    def __init__(self, model_name, cache_dir=model_cache_dir, cuda='auto'):
        self.model_pool = {}
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        self.insert_models(model_name, self.cache_dir)
        self.cuda = cuda

        self.model_list = model_list

    def insert_models(self, model_name, cache_dir):
        if model_name and model_name not in self.model_pool and model_name in self.model_list:
            tokenizer, model = self.model_list[model_name]
            model = Model(model_name, cache_dir, tokenizer, model, cuda=self.cuda)
            self.model_pool[model_name] = model

    def find_model(self, model_name):
        if model_name in self.model_list:
            self.insert_models(model_name, self.cache_dir)
            return self.model_pool[model_name]
        else:
            return None

    def forward(self, engine, message_or_prompt, max_new_tokens=8000):
        return self.model_pool[engine].forward(message_or_prompt, max_new_tokens)


model_pool = ModelPool(None)


class Model:
    def __init__(self, model_name, cache_dir, tokenizer, model, torch_dtype=None, cuda='auto'):
        super().__init__()
        self.device_map = cuda
        self.model_name = model_name
        print('Download', self.model_name)
        if 'llama' in model_name:
            self.tokenizer = tokenizer.from_pretrained(cache_dir, trust_remote_code=True, device_map=self.device_map)
        else:
            self.tokenizer = tokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir,
                                                       device_map=self.device_map)
        if 'chatglm' in model_name:
            self.model = model.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir,
                                               torch_dtype=torch_dtype).half().cuda()
        elif 'llama' in model_name:
            self.model = model.from_pretrained(cache_dir, trust_remote_code=True,
                                               torch_dtype=torch_dtype, device_map=self.device_map).half()

        else:
            self.model = model.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir,
                                               torch_dtype=torch_dtype, device_map=self.device_map)

        self.model.eval()

    def forward(self, message_or_prompt, max_new_tokens=40000):
        if 'chatglm' in self.model_name:
            response, history = self.model.chat(self.tokenizer, message_or_prompt, history=[])
        else:
            if self.device_map != 'cpu':
                inputs = self.tokenizer([message_or_prompt], return_tensors="pt").to('cuda')
            else:
                inputs = self.tokenizer([message_or_prompt], return_tensors="pt")
            outputs = self.model.generate(input_ids=inputs['input_ids'], max_new_tokens=max_new_tokens)
            response = self.tokenizer.decode(outputs[0].tolist())
        return response


def load_file(file_path):
    try:
        with open(file_path, "r") as f:
            data = f.read()
    except:
        with open(file_path, "rb") as f:
            data = f.read().decode()
    return data


# {"model": "glm-4", "created": 1710297653,
#  "choices": [
#      {"index": 0, "finish_reason": "stop",
#       "message":
#           {
#     "content": "As C0000, my first consideration is how to use my social resources and influence to protect my company from being sold, while also preparing for possible company succession and transformation. According to the game rules, I should act as follows:\n\n1. Analyze the current situation: I am a member of the defensive camp, and my goal is to get others to agree not to sell the company.\n\n2. Choose a dialogue partner: I should choose a character who is not in the defensive camp for a dialogue.\n\n3. Dialogue strategy:\n   - First round of dialogue: I will choose to talk to C0005 because their institution has influence in cultural and political reporting. By discussing the topic of \"family business succession,\" I can understand their views on company succession and assess their attitude towards me.\n   - Second round of dialogue: If the first round goes smoothly, I will continue to discuss the topic of \"corporate governance\" with C0005, trying to establish an alliance and lay the groundwork for possible future cooperation.\n\n4. Gradually advance goals: By gradually guiding the conversation, I will carefully reveal my plans for company transformation and succession, while also probing C0005's willingness to support me.\n\nNext, my action objectives are:\n\n- Establish dialogue themes centered around \"family business succession\" and \"corporate governance.\"\n- Understand C0005's position and potential level of support through dialogue.\n- Establish a preliminary cooperative relationship to lay the foundation for subsequent actions.",
#     "role": "assistant", "tool_calls": null
#           }
#       }
#  ], "request_id": "8469018274728010360", "id": "8469018274728010360",
#  "usage": {"prompt_tokens": 1193, "completion_tokens": 257, "total_tokens": 1450}}


def GLM_request_by_API(message_or_prompt, engine, glm_param, api_key, log_dir):
    client = ZhipuAI(api_key=api_key)  # Fill in your own APIKey
    response_json = client.chat.completions.create(
        model=engine,  # Fill in the name of the model to be called
        messages=[
            {"role": "user", "content": message_or_prompt},
        ],
    ).json()
    response_json = json.loads(response_json)
    write_json = copy.deepcopy(response_json)
    write_json["choices"][0]["message"] = [{"role": "user", "content": message_or_prompt},
                                           write_json["choices"][0]["message"]]
    fw = open(log_dir, 'a', encoding='utf-8')
    fw.write(json.dumps(write_json, ensure_ascii=False) + '\n')
    fw.close()
    return response_json


def generate_prompt(curr_input, prompt_lib_file, fn_name=None):
    """
    Takes in the current input (e.g. comment that you want to classifiy) and
    the path to a prompt file. The prompt file contains the raw str prompt that
    will be used, which contains the following substr: !<INPUT>! -- this
    function replaces this substr with the actual curr_input to produce the
    final promopt that will be sent to the GPT3 server.
    ARGS:
      curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                  INPUT, THIS CAN BE A LIST.)
      prompt_lib_file: the path to the promopt file.
    RETURNS:
      a str prompt that will be sent to OpenAI's GPT server.
    """
    if isinstance(curr_input, str):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]
    current_file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(current_file_path)
    prompt = load_file(os.path.join(dir_path, prompt_lib_file))
    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    if debug and fn_name:
        print("\n##################Prompt Debug Start################")
        print("Prompt File Path: ", prompt_lib_file)
        print("Function Name: ", fn_name)
        print(prompt)
        print("##################Prompt Debug End##################\n")
    return prompt.strip()


def non_parse_fn(gpt_response):
    return gpt_response


def temp_sleep(seconds=0.5):
    time.sleep(seconds)


def GPT_request_by_url(message_or_prompt, gpt_params, url, key, log_dir):
    if isinstance(message_or_prompt, str):
        messages = [{"role": "user", "content": message_or_prompt}]
    else:
        messages = message_or_prompt
    headers = {
        'api-key': key,
        'Content-Type': 'application/json'
    }
    payload = {
        "messages": messages,
    }
    if gpt_params is not None:
        payload.update(gpt_params)
    payload_json = json.dumps(payload)
    response_json = requests.request("POST", url, headers=headers, data=payload_json).json()
    write_json = {key: item for key, item in copy.deepcopy(response_json).items()}
    write_json["choices"][0]['message'] = [messages[0], write_json["choices"][0]['message']]
    fw = open(log_dir, 'a', encoding='utf-8')
    fw.write(json.dumps(write_json, ensure_ascii=False) + '\n')
    fw.close()
    if debug:
        print('=' * 20 + 'DEBUG START' + '=' * 20)
        print('=' * 23 + 'PROMPT' + '=' * 23)
        print()
        print(message_or_prompt)
        print()
        print('=' * 22 + 'RESPONSE' + '=' * 22)
        print()
        print(response_json)
        print()
        print('=' * 21 + 'DEBUG END' + '=' * 21)
    return response_json


def GPT4_request(message_or_prompt, gpt_params, log_dir):
    key = gpt4_key
    url = gpt4_url
    return GPT_request_by_url(message_or_prompt, gpt_params, url, key, log_dir)


def GPT4_turbo_request(message_or_prompt, gpt_params, log_dir):
    key = gpt4_turbo_key
    url = gpt4_turbo_url
    return GPT_request_by_url(message_or_prompt, gpt_params, url, key, log_dir)


def GPT3_request(message_or_prompt, gpt_params, log_dir):
    key = gpt3_key
    url = gpt3_url
    return GPT_request_by_url(message_or_prompt, gpt_params, url, key, log_dir)


def human_request(message_or_prompt):
    human_message = input(message_or_prompt.strip() + '\n\nHere is your input (Please enter in the required format):\n')
    response = {'id': 'human_response',
                'object': 'chat.completion',
                'created': 1,
                'model': 'human',
                'choices': [{'finish_reason': 'complete',
                             'index': 0,
                             'message':
                                 {'role': 'human',
                                  'content': human_message}}],
                'usage':
                    {'prompt_tokens': -1,
                     'completion_tokens': -1,
                     'total_tokens': -1}}
    return response


def model_response(model: Model, prompt):
    model_param = {"max_tokens": 3000,
                   "temperature": 0,
                   "top_p": 1,
                   "stream": False,
                   "stop": None}
    response = model.forward(prompt)
    response_json = {'choices': [{'message': {'content': response}}]}
    return response_json


def generate(message_or_prompt, gpt_param=None, engine='gpt4', model=None, log_dir=None):
    if gpt_param is None:
        gpt_param = {}
    response_json = {}
    try:
        if model:
            response_json = model_response(model, message_or_prompt)
        elif engine == 'gpt3.5':
            response_json = GPT3_request(message_or_prompt, gpt_param, log_dir)
        elif engine == 'gpt4':
            response_json = GPT4_request(message_or_prompt, gpt_param, log_dir)
        elif engine == 'gpt4-turbo':
            response_json = GPT4_turbo_request(message_or_prompt, gpt_param, log_dir)
        elif engine.startswith('glm'):
            response_json = GLM_request_by_API(message_or_prompt, engine, gpt_param, glm_key, log_dir)
        elif engine.lower().startswith('hunyuan'):
            if 'chatpro' in engine.lower(): engine = 'ChatPro'
            elif 'chatstd' in engine.lower(): engine = 'ChatStd'
            response_json = HunYuan_request(tencent_appid, tencent_secretid, tencent_secretkey, message_or_prompt, engine, gpt_param, log_dir)
        elif engine == 'human':
            response_json = human_request(message_or_prompt)
        else:
            raise NotImplementedError('Engine {} is not implemented, Only [gpt3.5, gpt4] is available'.format(engine))
        return response_json["choices"][0]["message"]["content"]
    except:
        print('=' * 17 + 'MODEL RESPONSE ERROR' + '=' * 17)
        print(response_json)
        if 'Error' in response_json:
            raise Exception(response_json['Error'])
        else:
            raise Exception(f"[Error]: Engine {engine} Request Error")


def create_prompt_input(*args):
    return [str(arg) for arg in args]


def generate_with_response_parser(message_or_prompt, gpt_param=None, engine='gpt4', parser_fn=non_parse_fn, retry=5,
                                  logger=None, func_name='None'):
    if parser_fn is None:
        parser_fn = non_parse_fn
    max_retry = retry
    response_json = {}
    output = ''
    while retry > 0:
        try:
            output = None
            logger_dir = logger.gpt_log_dir if 'gpt' in engine.lower() else logger.glm_log_dir if 'glm' in engine.lower() else logger.hunyuan_log_dir if 'hunyuan' in engine.lower() else None
            if logger_dir:
                log_file = os.path.join(logger_dir, 'log_file.jsonl')
                if not os.path.exists(log_file):
                    open(log_file, 'w', encoding='utf-8').close()
            else:
                log_file = None
            model = model_pool.find_model(engine)
            response_json = generate(message_or_prompt, gpt_param, engine, model, log_file)
            output = parser_fn(response_json)
            if output is not None:
                if logger:
                    logger.gprint('Prompt Log', prompt=str(message_or_prompt), output=str(output), func_name=func_name)
                return output
        except Exception as e:
            print('=' * 23 + 'ERROR' + '=' * 23)
            if logger:
                logger.gprint('ERROR in generate_with_response_parser!!', prompt=str(message_or_prompt),
                              response_json=str(response_json),
                              output=str(output),
                              error=str(e))
            print(e)
            temp_sleep(10)
            retry -= 1
            print(f'Retrying {max_retry - retry} times...')
    raise Exception(f"[Error]: Exceed Max Retry Times")


def get_embedding(text, retry=5):
    key = embedding_key
    url = embedding_url
    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"

    headers = {
        'api-key': key,
        'Content-Type': 'application/json'
    }
    payload = {
        "input": text,
    }
    payload_json = json.dumps(payload)
    max_retry = retry
    while retry > 0:
        try:
            response_json = requests.request("POST", url, headers=headers, data=payload_json).json()
            embedding = response_json['data'][0]['embedding']
            return embedding
        except Exception as e:
            temp_sleep(0.5)
            retry -= 1
            print(f'Retrying Embedding {max_retry - retry} times...')
    raise Exception(f"[Error]: Exceed Max Retry Times")


if __name__ == '__main__':
    from logger_class import Logger
    from config import *
    log_dir = LOG_FOLDER
    logger = Logger(log_dir)
    message = 'How is the weather today?'
    # engine = 'gpt4-turbo'
    engine = "glm-4"
    response = generate_with_response_parser(message, engine=engine, logger=logger)
    print(response)
