from prompt.gpt_structure import generate_prompt, generate_with_response_parser, create_prompt_input
import sys
import re


def run_speech(character_id_number,
               character_description,
               action_history_description,
               candidates_description,
               resources,
               support_character,
               engine='gpt4',logger=None):
    gpt_param = {"temperature": 0.3, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}

    prompt_template = "prompt_files/prompt_4_speech.txt"
    # prompt_template = "prompt_files/prompt_wo_thinking/prompt_4_speech_wo_thinking.txt"
    prompt_input = create_prompt_input(character_id_number,
                                       character_description,
                                       action_history_description,
                                       candidates_description,
                                       resources,
                                       support_character)
    prompt = generate_prompt(prompt_input, prompt_template, fn_name=sys._getframe().f_code.co_name)

    def parse_output(gpt_response):
        try:
            reasoning_process = gpt_response.split('### Reasoning Process:')[-1].split('### Speech:')[0].strip()
            speech = gpt_response.split('### Speech:')[-1]
        except:
            print('ERROR\nERROR\n', gpt_response, '\nERROR\nERROR\n')
            raise Exception("[Error]: GPT response parse error")
        return speech, reasoning_process

    if engine == 'human':

        human_prompt = 'As %s, %s\nPlease make some declarations to all other characters:\n' % (
                                                            character_id_number,
                                                            character_description)
        logger.gprint(human_prompt,
                      thought=human_prompt,
                      important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Human Speaking',
                      log_content=human_prompt)

        reasoning_process = 'This is a human; no reasoning process is needed.'
        speech = generate_with_response_parser(human_prompt, engine=engine, logger=logger,func_name='human_speech')


        logger.gprint(human_prompt,
                      thought=human_prompt,
                      important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Human Speaking Result',
                      log_content=speech)
    else:
        speech, reasoning_process = generate_with_response_parser(prompt, gpt_param=gpt_param, parser_fn=parse_output, engine=engine,
                                                  logger=logger,func_name='run_speech')
    speech = speech.replace('\n', ' ').strip()
    return speech, reasoning_process

def run_groupchat(character_id_number,
               character_description,
               action_history_description,
               candidates_description,
               resources,
               round_description,
               support_character,
               engine='gpt4',
               logger=None):
    gpt_param = {"temperature": 0.3, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}

    if not support_character:
        support_character = 'You currently do not support any other characters, so you only consider your own interests.'
    prompt_template = "prompt_files/prompt_4_groupchat.txt"
    prompt_input = create_prompt_input(character_id_number,
                                       character_description,
                                       action_history_description,
                                       candidates_description,
                                       resources,
                                       support_character,
                                       round_description)
    prompt = generate_prompt(prompt_input, prompt_template, fn_name=sys._getframe().f_code.co_name)

    def parse_output(gpt_response):
        try:
            reasoning_process = gpt_response.split('### Reasoning Process')[-1].split('### Speech:')[0].strip()
            speech = gpt_response.split('### Speech:')[-1]
        except:
            print('ERROR\nERROR\n', gpt_response, '\nERROR\nERROR\n')
            raise Exception("[Error]: GPT response parse error")
        return speech, reasoning_process

    if engine == 'human':
        reasoning_process = 'This is a human; no reasoning process is needed.'
        speech = generate_with_response_parser('As %s, please post some group chat messages to all other characters:\n' % (character_id_number),
                                               engine=engine, logger=logger,func_name='human_speech_round')
    else:
        speech, reasoning_process = generate_with_response_parser(prompt, gpt_param=gpt_param, parser_fn=parse_output, engine=engine,
                                                  logger=logger,func_name='run_speech_round')
    speech = speech.replace('\n', ' ').strip()
    return speech, reasoning_process