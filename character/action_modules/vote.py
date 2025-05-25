from prompt.gpt_structure import generate_prompt, generate_with_response_parser, create_prompt_input
import sys


def run_vote(source_character_id_number: str,
             source_character_description: str,
             self_belief_description: str,
             vote_requirement: str,
             background_information: str,
             candidates: str,
             support_character: str,
             engine='gpt4',
             requirement_list=None,
             logger=None):
    '''
    Character casts a vote.
    Input:
        source_character_id_number: str,
        source_character_description:str,
        self_belief_description: str, 
        vote_requirement: str voting requirements.
        background_information: str background information.
        candidates: str entities that can be voted for.

    Output:
        choice : str, ID number of the chosen character(s).
        reasoning_process: str, reasoning/justification.
    '''

    gpt_param = {"temperature": 0.3, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}

    if not support_character:
        support_character = 'You currently do not support any other characters, so you only consider your own interests.'

    prompt_template = "prompt_files/prompt_4_vote.txt"
    prompt_input = create_prompt_input(source_character_id_number,
                                       source_character_description,
                                       self_belief_description,
                                       vote_requirement,
                                       background_information,
                                       candidates,
                                       support_character)
    prompt = generate_prompt(prompt_input, prompt_template, fn_name=sys._getframe().f_code.co_name)

    def parse_output(gpt_response):
        ret = {}
        try:
            # reasoning_process = re.search(r"### Reasoning Process:(.*)\n", gpt_response).group(1).strip()
            # choice = re.search(r"### Choice:(.*)",gpt_response).group(1).strip()
            action_space = [i.strip() for i in gpt_response.split('### Action Space:')[-1].split('### Reasoning Process')[0].strip().split(',')]
            reasoning_process = gpt_response.split('### Reasoning Process:')[-1].split('### Choice:')[0].strip()
            choice = [i.strip() for i in gpt_response.split('### Choice:')[-1].strip().split(',')]
            ret['reasoning_process'] = reasoning_process
            ret['choice'] = choice

            if not choice or not reasoning_process:
                raise Exception("[Error]: GPT response not in given format")
        except:
            print('ERROR\nERROR\n', gpt_response, '\nERROR\nERROR\n')
            raise Exception("[Error]: GPT response parse error")
        return action_space, choice, reasoning_process

    if engine == 'human':
        action_space='[SKIP]'
        human_prompt = 'You are %s, your description is %s.\nCandidate list:\n%s\nPlease enter the ID of the character you want to vote for:\n' % (source_character_id_number,
                                                                                    source_character_description,
                                                                                    candidates)
        logger.gprint(human_prompt,
                      thought=human_prompt,
                      requirement=requirement_list,
                      important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Human Choosing',
                      log_content=human_prompt)
        choice=generate_with_response_parser(human_prompt,
                                             engine=engine,
                                             logger=logger,func_name='human_vote')
        logger.gprint(choice,
                      thought=human_prompt,
                      requirement=requirement_list,
                      important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Human Choosing Result',
                      log_content=choice)
        choice = [choice, choice]
        reasoning_process='This is a human; no reasoning is needed.'
    else:
        action_space, choice, reasoning_process = generate_with_response_parser(prompt,
                                                                                gpt_param=gpt_param,
                                                                                parser_fn=parse_output,
                                                                                engine=engine,
                                                                                logger=logger,func_name='run_vote')
    return action_space, choice, reasoning_process


