'''
Code copied from Generative Agents
'''
import time

import os
import copy
from ctransformers import AutoModelForCausalLM # Added import
from zhipuai import ZhipuAI

from config import *
from .utils import *
from .hunyuan import HunYuan_request
# Import GGUF model configurations
from config import AVAILABLE_GGUF_MODELS, GGUF_MODELS_DIR


class ModelPool:
    def __init__(self, model_name, cache_dir=model_cache_dir, cuda='auto'):
        self.model_pool = {}
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir
        # Potentially load initial models if model_name is not None or a GGUF path
        if model_name and not model_name.endswith(".gguf"):
            self.insert_models(model_name, self.cache_dir)
        elif model_name and model_name.endswith(".gguf"): # Handle GGUF path at init
            self.insert_models(model_name, self.cache_dir)
        self.cuda = cuda
        self.model_list = model_list # Keep for HuggingFace models

    def insert_models(self, model_identifier, cache_dir): # model_identifier can be name or path
        # If model_identifier is a short name for a GGUF model
        if model_identifier in AVAILABLE_GGUF_MODELS:
            actual_model_path = AVAILABLE_GGUF_MODELS[model_identifier]
            # Ensure the path exists, GGUF_MODELS_DIR should make it absolute or correctly relative
            # The key in model_pool should be the actual_model_path for uniqueness
            if actual_model_path not in self.model_pool:
                if not os.path.exists(actual_model_path):
                    print(f"Error: GGUF model file not found at path (from short name '{model_identifier}'): {actual_model_path}")
                    return
                try:
                    print(f"Loading GGUF model (from short name '{model_identifier}'): {actual_model_path}")
                    gguf_model = AutoModelForCausalLM.from_pretrained(actual_model_path)
                    # Use actual_model_path as the key in model_pool and for Model's model_name
                    self.model_pool[actual_model_path] = Model(actual_model_path, None, None, gguf_model, model_type='gguf')
                    print(f"GGUF model {actual_model_path} loaded successfully.")
                except Exception as e:
                    print(f"Error loading GGUF model {actual_model_path} (from short name '{model_identifier}'): {e}")
        # If model_identifier is a direct GGUF file path
        elif model_identifier.endswith(".gguf"):
            if model_identifier not in self.model_pool:
                if not os.path.exists(model_identifier):
                    print(f"Error: GGUF model file not found at direct path: {model_identifier}")
                    return
                try:
                    print(f"Loading GGUF model (direct path): {model_identifier}")
                    gguf_model = AutoModelForCausalLM.from_pretrained(model_identifier)
                    self.model_pool[model_identifier] = Model(model_identifier, None, None, gguf_model, model_type='gguf')
                    print(f"GGUF model {model_identifier} loaded successfully.")
                except Exception as e:
                    print(f"Error loading GGUF model {model_identifier} (direct path): {e}")
        # If model_identifier is a HuggingFace model name from model_list
        elif model_identifier and model_identifier not in self.model_pool and model_identifier in self.model_list:
            tokenizer, model_loader = self.model_list[model_identifier]
            model_instance = Model(model_identifier, cache_dir, tokenizer, model_loader, cuda=self.cuda)
            self.model_pool[model_identifier] = model_instance
        # else:
            # This 'else' might be hit if a name is passed that isn't in any category yet,
            # find_model will handle returning None if it's not found after checks.
            # print(f"Model identifier '{model_identifier}' not recognized for insertion immediately.")


    def find_model(self, model_name_or_path):
        # Priority 1: Check if it's a short name for a GGUF model
        if model_name_or_path in AVAILABLE_GGUF_MODELS:
            actual_model_path = AVAILABLE_GGUF_MODELS[model_name_or_path]
            if actual_model_path not in self.model_pool:
                self.insert_models(model_name_or_path, None) # Pass short name, insert_models resolves it
            # The key in model_pool is actual_model_path
            return self.model_pool.get(actual_model_path) 
        
        # Priority 2: Check if it's a direct .gguf file path
        elif model_name_or_path.endswith(".gguf"):
            if model_name_or_path not in self.model_pool:
                self.insert_models(model_name_or_path, None) # Pass direct path
            return self.model_pool.get(model_name_or_path)

        # Priority 3: Check if it's a HuggingFace model from model_list
        elif model_name_or_path in self.model_list:
            if model_name_or_path not in self.model_pool:
                self.insert_models(model_name_or_path, self.cache_dir) # Pass HF model name
            return self.model_pool.get(model_name_or_path)
        
        # If not found in any category
        else:
            print(f"Model '{model_name_or_path}' not found in AVAILABLE_GGUF_MODELS, as a direct GGUF path, or in predefined HuggingFace model list.")
            return None

    def forward(self, engine, message_or_prompt, max_new_tokens=8000): # This method might be redundant if Model.forward is used directly
        model = self.find_model(engine)
        if model:
            return model.forward(message_or_prompt, max_new_tokens)
        else:
            # Fallback or error for unknown engine
            raise ValueError(f"Model for engine '{engine}' not found.")


model_pool = ModelPool(None) # Initialize with no specific model initially


class Model:
    def __init__(self, model_name, cache_dir, tokenizer_class, model_class, torch_dtype=None, cuda='auto', model_type=None):
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type # Store model_type ('gguf' or None)

        if self.model_type == 'gguf':
            self.model = model_class # This is already the loaded ctransformers model object
            self.tokenizer = None # GGUF models handle tokenization internally
            print(f"GGUF model '{model_name}' initialized.")
        else: # Existing HuggingFace model loading logic
            self.device_map = cuda
            print('Download', self.model_name)
            # Correctly use tokenizer_class and model_class which are expected to be classes for .from_pretrained
            if 'llama' in model_name: # This special handling for llama might need review
                self.tokenizer = tokenizer_class.from_pretrained(cache_dir, trust_remote_code=True, device_map=self.device_map)
                self.model = model_class.from_pretrained(cache_dir, trust_remote_code=True,
                                                          torch_dtype=torch_dtype, device_map=self.device_map).half()

            else:
                self.tokenizer = tokenizer_class.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir,
                                                                  device_map=self.device_map)
                if 'chatglm' in model_name:
                    self.model = model_class.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir,
                                                              torch_dtype=torch_dtype).half().cuda() # Specific GLM settings
                else:
                    self.model = model_class.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir,
                                                              torch_dtype=torch_dtype, device_map=self.device_map)
            self.model.eval()

    def forward(self, message_or_prompt, max_new_tokens=40000): # max_new_tokens might not be used by ctransformers directly
        if hasattr(self, 'model_type') and self.model_type == 'gguf':
            try:
                # ctransformers model call. Parameters like max_new_tokens might need to be passed if supported.
                # For simplicity, using basic call first.
                # response = self.model(message_or_prompt, max_new_tokens=max_new_tokens) # If ctransformers supports it directly
                response = self.model(message_or_prompt)

                # The ctransformers llm call often returns a generator for streaming.
                # If it's not streaming, it might return the full string.
                # Assuming it returns the full string here for simplicity based on common usage.
                # If it's a generator, you'd iterate: full_response = "".join(token for token in response)
                return response
            except Exception as e:
                print(f"Error during GGUF model forward pass: {e}")
                return "Error generating response from GGUF model."
        elif 'chatglm' in self.model_name: # Existing logic for ChatGLM
            response, history = self.model.chat(self.tokenizer, message_or_prompt, history=[])
        else: # Existing logic for other HuggingFace models
            if self.device_map != 'cpu':
                inputs = self.tokenizer([message_or_prompt], return_tensors="pt").to('cuda' if self.device_map == 'auto' else self.device_map)
            else:
                inputs = self.tokenizer([message_or_prompt], return_tensors="pt")
            outputs = self.model.generate(input_ids=inputs['input_ids'], max_new_tokens=max_new_tokens)
            response = self.tokenizer.decode(outputs[0].tolist()) # Make sure to decode only the generated part if needed
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
    if isinstance(message_or_prompt, str): # Ensure messages is a list of dicts
        messages = [{"role": "user", "content": message_or_prompt}]
    elif isinstance(message_or_prompt, list) and all(isinstance(m, dict) for m in message_or_prompt):
        messages = message_or_prompt # Already in correct format
    else:
        # Attempt to convert if it's a different format, or raise an error
        raise ValueError("message_or_prompt must be a string or a list of message dictionaries.")
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
    # For GGUF models, model.forward directly returns the text response.
    if hasattr(model, 'model_type') and model.model_type == 'gguf':
        response = model.forward(prompt) # prompt is message_or_prompt
    else: # For other models, it might return a more complex object or already be text
        response = model.forward(prompt) # Assuming HF model.forward also returns text or is handled by it

    response_json = {'choices': [{'message': {'content': response}}]}
    return response_json


def generate(message_or_prompt, gpt_param=None, engine='gpt4', model=None, log_dir=None):
    if gpt_param is None:
        gpt_param = {}
    response_json = {}
    try:
        if model: # This 'model' is an instance of our Model class
            # Pass gpt_param if ctransformers/HF models need it (e.g. max_tokens)
            # For GGUF, max_new_tokens is passed to forward in model_response
            # For HF, max_new_tokens is also handled by Model.forward
            if hasattr(model, 'model_type') and model.model_type == 'gguf':
                response_json = model_response(model, message_or_prompt)
            else: # Existing HuggingFace model or other types handled by Model class
                response_json = model_response(model, message_or_prompt)
        # The following elif for engine.endswith(".gguf") might be redundant if find_model correctly handles paths.
        # find_model (called in generate_with_response_parser) should already resolve GGUF paths and short names.
        # If 'model' is None here, it means find_model failed.
        # The `else` block below handles cases where `model` is None (i.e., engine not found by find_model).
        # elif engine.endswith(".gguf"): # This specific check might be redundant
        #     print(f"Attempting to load GGUF model directly in generate: {engine}")
        #     model_instance = model_pool.find_model(engine) # find_model now handles short names too
        #     if model_instance:
        #         response_json = model_response(model_instance, message_or_prompt)
        #     else:
        #         raise FileNotFoundError(f"GGUF model {engine} not found or failed to load (called from generate).")
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
        else: # This 'else' is hit if 'model' was None (engine not found by find_model) and not an API call.
              # find_model would have already tried resolving GGUF short names, paths, and HF models.
            raise NotImplementedError(
                f"Engine '{engine}' is not a recognized API, a direct GGUF path, "
                f"a registered GGUF short name in AVAILABLE_GGUF_MODELS, "
                f"or a pre-defined HuggingFace model in model_list."
            )
        return response_json["choices"][0]["message"]["content"]
    except Exception as e: # Catch specific exceptions if possible
        print('=' * 17 + 'MODEL RESPONSE ERROR' + '=' * 17)
        print(f"Error details: {e}")
        print(f"Response JSON at error: {response_json}")
        # Avoid raising exception with response_json['Error'] if response_json is not a dict or 'Error' key doesn't exist
        if isinstance(response_json, dict) and 'Error' in response_json:
            raise Exception(response_json['Error'])
        else:
            raise Exception(f"[Error]: Engine {engine} Request Error - {str(e)}")


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
            model = None # Initialize model to None
            # Determine logger directory
            if logger:
                log_file_name = 'log_file.jsonl'
                if 'gpt' in engine.lower(): logger_dir = logger.gpt_log_dir
                elif 'glm' in engine.lower(): logger_dir = logger.glm_log_dir
                elif 'hunyuan' in engine.lower(): logger_dir = logger.hunyuan_log_dir
                # For GGUF models (short name or path), use a dedicated log or default
                elif engine in AVAILABLE_GGUF_MODELS or engine.endswith(".gguf"):
                    logger_dir = getattr(logger, 'gguf_log_dir', os.path.join(logger.log_dir, 'gguf_logs'))
                else: # Default for other engines or if engine name is unusual
                    logger_dir = os.path.join(logger.log_dir, 'other_model_logs')
                
                log_file = os.path.join(logger_dir, log_file_name)
                if not os.path.exists(logger_dir): os.makedirs(logger_dir) # Ensure log directory exists
                if not os.path.exists(log_file): open(log_file, 'w', encoding='utf-8').close()
            else:
                log_file = None

            # Model loading is now centralized in find_model, which handles GGUF short names, paths, and HF models.
            # API calls are handled directly in `generate` if `model` is None.
            model = model_pool.find_model(engine) 
            # `model` will be None if `engine` is an API type (e.g., 'gpt4') or if the model isn't found.
            # `generate` handles the case where `model` is None for API calls.
            # If `model` is None and it's not an API call, `generate` will raise NotImplementedError.

            response_content_string = generate(message_or_prompt, gpt_param, engine, model, log_file)
            output = parser_fn(response_content_string) # parser_fn expects the content string directly
            if output is not None:
                if logger:
                    # Ensure prompt and output are strings for logging
                    log_prompt = str(message_or_prompt)
                    log_output = str(output)
                    logger.gprint('Prompt Log', prompt=log_prompt, output=log_output, func_name=func_name)
                return output
        except Exception as e:
            print('=' * 23 + 'ERROR in generate_with_response_parser' + '=' * 23)
            error_context = {
                "prompt": str(message_or_prompt),
                "engine": engine,
                "gpt_param": gpt_param,
                "response_at_error": response_json if 'response_json' in locals() else "N/A", # response_json might not be defined if error is early
                "current_output_attempt": str(output),
                "error_message": str(e)
            }
            print(error_context)
            if logger:
                logger.gprint('ERROR in generate_with_response_parser!!', **error_context) # Log the context
            
            temp_sleep(10) # Consider making sleep duration configurable or dynamic
            retry -= 1
            if retry > 0:
                 print(f'Retrying... {retry} attempts left.')
            else:
                 print('Max retries exceeded.')
    raise Exception(f"[Error]: Exceed Max Retry Times for engine {engine} after {max_retry} attempts. Last error: {str(e if 'e' in locals() else 'Unknown error')}")


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
