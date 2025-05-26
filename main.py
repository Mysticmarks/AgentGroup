'''
Overall Process
1. Initialize Character, Environment, factions
2. Competition phase
3. Cooperation phase
4. Reflection phase
5. Settlement phase
'''
import collections
import json

from character.all_character_class import AllCharacter
from environment.all_resource_class import AllResource
from environment.action_history_class import ActionHistory, Action
from character.character_class import Character # Import Character class

import os
from config import *
from logger_class import Logger

def verify_constrained_action(gpt_response, action_candidates:list)->bool:
    action_candidates = [str(i) for i in action_candidates]
    gpt_response = str(gpt_response)
    if debug:
        print('='*20+'DEBUG START'+'='*20)
        print('='*19+'VERIFICATION'+'='*19)
        print()
        print(action_candidates)
        print()
        print('='*19+'GPT RESPONSE'+'='*19)
        print()
        print(gpt_response)
        print()
        print('='*15+'VERIFICATION RESULT'+'='*15)
        print()
        print(gpt_response in action_candidates)
        print()
        print('='*21+'DEBUG END'+'='*21)
    if gpt_response not in action_candidates:
        return False
    else:
        return True

def succession_winner(defender_id_number, character_vote_dict)->list:
    defender_chosen_id_number = character_vote_dict[defender_id_number]
    if defender_chosen_id_number != defender_id_number:
        return defender_chosen_id_number
    vote_list = []
    for key, vote_for in character_vote_dict.items():
        vote_list.append(vote_for)
    vote_dict = collections.Counter(vote_list)
    winner = []
    winner_get_vote = -1
    for character_id_number, get_vote in vote_dict.items():
        if get_vote > winner_get_vote:
            winner_get_vote = get_vote
            winner = []
        winner.append(character_id_number)
    if type(winner) != list:
        winner = [winner]
    return winner


class AgentGroupChat:
    def __init__(self,
                 all_round_number: int,
                 private_chat_round: int = 3,
                 meeting_chat_round: int = 3,
                 group_chat_round: int = 3,
                 save_folder=None,
                 test_folder=None,
                 human_input=None,
                 logger=None):
        '''
        Initializes the game environment.
        Input:
            all_round_number: int, total number of game rounds.
            private_chat_round: int, total number of dialogue rounds in each confrontation phase.
            meeting_chat_round: int, total number of dialogue rounds in each cooperation phase.
            save_folder: str, archive address.
            human_input: str, human input.
            logger: Logger, whether a Logger needs to be directly input.
        Output:
            None
        '''

        self.all_round_number = all_round_number
        self.private_chat_round = private_chat_round
        self.meeting_chat_round = meeting_chat_round
        self.group_chat_round = group_chat_round
        if not logger:
            self.logger = Logger()
        else:
            self.logger = logger
        self.log_file_name = self.logger.log_file

        self.save_folder = save_folder
        self.test_folder = test_folder
        if save_folder:
            self.initialize(save_folder)

            # Assign social influence to NPCs.
            for index, resource in enumerate(self.resources.get_all_resource()):
                owner_id_number = resource.owner
                self.characters.get_character_by_id(owner_id_number).give_influence(resource.influence)

        else:
            self.characters = AllCharacter(logger=self.logger)
            self.resources = AllResource()
            self.rule_setting = ''
            self.action_history = ActionHistory


    def initialize(self, save_folder) -> None:
        '''
        Initializes Logger, Character, Resources, RuleSetting, ActionHistory.
        Input:
            save_folder: Location where archive data is stored.
        Output:
            None
        '''
        self.logger.gprint('### Initializing Directory Found: ', save_folder)
        basic_setting = json.load(open(os.path.join(save_folder, 'basic_setting.json'), encoding='utf-8'))

        self.rule_setting_file_name = basic_setting['rule_setting']
        self.finished_states = basic_setting['finished_states']
        if basic_setting['log_file_name']:
            self.logger.read_save_file(basic_setting['log_file_name'], False)

        self.characters = AllCharacter(os.path.join(save_folder, 'characters'), logger=self.logger)
        self.resources = AllResource(os.path.join(save_folder, 'resources'))
        self.rule_setting = open(self.rule_setting_file_name, encoding='utf-8').read()
        self.action_history = ActionHistory(os.path.join(save_folder, 'action_history'), os.path.join(save_folder, 'basic_setting.json'))


        self.logger.gprint('### Number of initialized roles: ', len(self.characters.get_all_characters()))
        self.logger.gprint('### Number of main roles: ', len(self.characters.main_characters_id_number))
        self.logger.gprint('### Number of initialized resources: ', len(self.resources.get_all_resource()))
        self.logger.gprint('### Number of initialized action history: ',
                           len(self.action_history.get_all_action_history()))
    def switch_state(self):
        self.state_index = 0
        all_state = ['']

    def save(self, save_folder) -> None:
        '''
        Saves the environment.
        Input:
            save_folder: storage address.
        Output:
            None
        '''
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        config_file = open('config.py', encoding='utf-8')
        basic_setting = {}
        for line in config_file:
            line = line.split('#')[0]
            if '=' in line:
                key, value = [i.strip() for i in line.split('=')]
                basic_setting['config_file_'+key] = value
        basic_setting['rule_setting'] = self.rule_setting_file_name
        basic_setting['finished_states'] = self.finished_states
        basic_setting['log_file_name'] = self.log_file_name

        save_characters_folder = os.path.join(save_folder, 'characters')
        save_resources_folder = os.path.join(save_folder, 'resources')
        save_action_history_folder = os.path.join(save_folder, 'action_history')

        open(os.path.join(save_folder, 'basic_setting.json'), 'w', encoding='utf-8').write(json.dumps(basic_setting,
                                                                                                      indent=4,
                                                                                                      ensure_ascii=False))
        self.logger.gprint('Save basic_setting.json to: ' + str(os.path.join(save_folder, 'basic_setting.json')))
        for character in self.characters.get_all_characters(): character.save(save_characters_folder)
        self.logger.gprint('Save self.character to: ' + str(save_characters_folder))
        for resource in self.resources.get_all_resource(): resource.save(save_resources_folder)
        self.action_history.save(save_action_history_folder)
        self.logger.gprint('Save self.action_history to: ' + str(save_action_history_folder))

    def new_character_insert(self,
                             id_name: str,
                             name: str,
                             objective: str,
                             scratch: str,
                             background: str,
                             engine: str,
                             character_type: str = "ai",
                             is_main_character: bool = False,
                             support_character: str = '',
                             beliefs: dict = None,
                             judgements: dict = None, # Note: Character class uses defaultdict for judgement
                             relations: dict = None,   # Note: Character class uses defaultdict for relation
                             portrait: str = None,
                             small_portrait: str = None):
        '''
        Creates a new character, saves its configuration, and adds it to the simulation.
        Input:
            Parameters to define the new character.
        Output:
            The created Character object, or None if creation failed.
        '''
        if not self.save_folder:
            self.logger.gprint("Error: `save_folder` is not set. Cannot save new character.", level="ERROR")
            return None

        if id_name in self.characters.character_dict:
            self.logger.gprint(f"Error: Character with ID_NAME '{id_name}' already exists.", level="ERROR")
            return None

        # --- Create Character Instance ---
        new_char = Character(id_number=id_name, logger=self.logger)
        new_char.name = name
        new_char.objective = objective
        new_char.scratch = scratch
        new_char.background = background
        new_char.engine = engine
        new_char.type = character_type
        new_char.main_character = is_main_character
        new_char.support_character = support_character

        # Optional dictionary attributes
        if beliefs is not None:
            new_char.belief = beliefs
        # For judgements and relations, Character class initializes them as defaultdicts.
        # If specific initial values are provided, they can be set.
        if judgements is not None:
            for key, value in judgements.items(): # Assuming judgements is a simple dict
                new_char.judgement[key] = value 
        if relations is not None:
            for key, value in relations.items(): # Assuming relations is a simple dict
                new_char.relation[key] = value

        # Default portrait paths (example placeholders, adjust as needed)
        # Instructions mentioned using C0000.json defaults, but those are not available to me.
        # Using generic placeholders.
        default_portrait_path = "./assets/portraits/default_character_portrait.png" 
        default_small_portrait_path = "./assets/small_portraits/default_character_small_portrait.png"
        
        new_char.portrait = portrait if portrait else default_portrait_path
        new_char.small_portrait = small_portrait if small_portrait else default_small_portrait_path
        
        # --- Save Character Configuration ---
        characters_save_path = os.path.join(self.save_folder, 'characters')
        if not os.path.exists(characters_save_path):
            os.makedirs(characters_save_path, exist_ok=True)
        
        try:
            new_char.save(characters_save_path) # Character.save handles filename creation
            self.logger.gprint(f"Character '{name}' (ID: {id_name}) configuration saved to {os.path.join(characters_save_path, id_name + '.json')}", level="INFO")
        except Exception as e:
            self.logger.gprint(f"Error saving character '{name}' (ID: {id_name}): {e}", level="ERROR")
            return None

        # --- Add Character to Active Session ---
        # The add_character method in AllCharacter handles adding to lists and dicts.
        self.characters.add_character(new_char, logger_passed=self.logger) 
        # add_character in AllCharacter class already logs the addition to simulation.

        self.logger.gprint(f"Character '{name}' (ID: {id_name}) successfully created and added to the simulation.", level="INFO")
        return new_char

    def new_resource_insert(self):
        '''
        Inserts a new resource.
        Input:
            xxx
        Output:
            xxx
        '''
        pass

    def new_action_insert(self, new_action: list, now_round_number: int):
        '''
        Inserts a new action.
        Input:
            new_action: list [source_character_id_number:str, target_character_id_number:str, action_type:str, action:str]
            now_round_number: int
        Output:
            None
        '''
        action = Action(-1, new_action[0], new_action[1], new_action[2], new_action[3], now_round_number)
        action_index = self.action_history.insert_action(action)
        return action_index

    def get_rule_setting(self):
        '''
        Returns the Rule Setting.
        Input:
            None
        Output:
            None
        '''
        return self.rule_setting

    def get_all_resource_description(self):
        '''
        Returns the description of all resources.
        Input:
            None
        Output:
            None
        '''
        return self.resources.get_description()

    def get_all_character_list(self):
        '''
        Returns the list of all characters.
        Input:
            None
        Output:
            None
        '''
        return self.characters.get_characters_description_except_some()

    def get_round_description(self, now_round_number: int, private=False, simple=False) -> str:
        '''
        Gets some descriptive information about the current round and total rounds.
        Input:
            now_round_number: int, which round the current game is in.
            private: bool, whether the current round is in the confrontation phase.
        Output:
            round_description: str
        '''
        round_description = ''
        round_description += 'The game takes a total of %d rounds.\n' % self.all_round_number
        round_description += 'The current game is played to round %d.\n' % (now_round_number + 1)
        if simple:
            return round_description
        if private:
            round_description += 'You are in the private chatting stage, the stage where you meet with anyone without anyone else knowing about it.\n'
        else:
            round_description += 'You are in the confidential meeting stage, the stage where what you meet with someone will be known to everyone, but they won\'t know what you talked about.\n'
        round_description += 'You\'ll talk to your chosen character for %d rounds per round.\n' % (
            self.private_chat_round if private else self.meeting_chat_round)
        return round_description

    def get_groupchat_round_description(self, now_round_number, now_chat_round):
        round_description = self.get_round_description(now_round_number, simple=True)
        round_description += 'You are in a group chat and what you say will be visible to all characters.\n'
        round_description += 'A total of %d rounds of group chat are taking place, and you are currently in the %d round.'%(self.group_chat_round, now_chat_round)

        return round_description
    def group_chatting_stage(self, now_round_number:int)->None:
        '''
        Enters the declaration phase.
        Input:
            now_round_number: int,
        Output:
            None
        '''
        # Introduction of all characters.
        candidates = ['%s: %s' % (character.get_id_number(), character.get_short_description()) for character in
                      self.characters.get_all_characters()]
        candidates = '\n'.join(candidates)

        # Content of one round of group chat, visible to everyone in the next round.
        round_action_history = collections.defaultdict(list)

        # Set up an extra loop to insert all actions into action history.
        for now_chat_round in range(self.group_chat_round+1):

            # Introduction to the current round.
            round_description = self.get_groupchat_round_description(now_round_number,
                                                                     now_chat_round=now_chat_round+1)
            # Put the content of the previous round of group chat into action history.
            if now_chat_round-1 in round_action_history:
                for new_action, character in round_action_history[now_chat_round-1]:
                    state_UID = 'NOW_ROUND:%d+ACTION:%s+CHARACTER:%s' % (now_round_number, 'ANNOUNCEMENT', character.id_number)
                    if state_UID in self.finished_states: continue
                    action_index = self.new_action_insert(new_action, now_round_number)
                    self.finished_states[state_UID] = [action_index]
                    if self.test_folder:
                        self.save(self.test_folder)

            # Terminate the extra loop.
            if now_chat_round >= self.group_chat_round: break

            # Main characters act sequentially according to their influence.
            for character in self.characters.character_list:

                action_history = self.action_history.get_description(character_id_number=character.id_number, max_num=ACTIONHISTORY_RETRIEVE_NUM_ANNOUNCEMENT)
                # ======================================================================================= #
                # Call GPT.
                # No validation needed.
                # ======================================================================================= #
                speech, reasoning_process = character.groupchat(action_history,
                                                                  candidates,
                                                                  self.resources.get_description(),
                                                                  round_description,
                                                                  )
                # ======================================================================================= #
                # Log it.
                speech = 'Game Round %d, Chat Round %d, group chat that character %s makes to all the other characters: %s' % (now_round_number+1, now_chat_round+1, character.id_number, speech)
                self.logger.gprint(thought=reasoning_process,
                    important_log='important_log',
                    source_character=character.id_number,
                    target_character=character.id_number,
                    log_type='Open Speech In Round',
                    log_content=speech)
                # Record action.
                new_action = [character.id_number, character.id_number, '### SPEECH_NORMAL', speech]
                round_action_history[now_chat_round].append((new_action, character))

    def private_chatting_stage(self, now_round_number: int) -> None:
        '''
        Confrontation phase—all MCs act sequentially according to their influence, choosing a character from a different faction for dialogue.
        Input:
            now_round_number: int, current round.
        Output:
            None
        '''
        round_description = self.get_round_description(now_round_number, private=True)

        main_character_influence = self.characters.get_main_character_influence()
        # Main characters act sequentially according to their influence.
        for main_character_id_number in main_character_influence:
            state_UID = 'NOW_ROUND:%d+ACTION:%s+CHARACTER:%s'%(now_round_number, 'COMPETE', main_character_id_number)
            if state_UID in self.finished_states: continue
            action_index = []
            # Get the main character taking action.
            main_character = self.characters.get_character_by_id(main_character_id_number)
            self.logger.gprint(thought='',
                               important_log='important_log',
                               source_character=main_character.id_number,
                               target_character=main_character.id_number,
                               log_type='Action stage',
                               log_content='Confrontation stage'
                               )

            main_character_action_history_description = self.action_history.get_description(main_character_id_number, max_num=ACTIONHISTORY_RETRIEVE_NUM_COMPETE)
            # ======================================================================================= #
            # Call GPT.
            # No validation needed.
            # ======================================================================================= #
            # Let the character perceive the environment and generate a summary.
            main_character_environment_summary = main_character.perceive(self.rule_setting,
                                                                         self.resources.get_description(),
                                                                         main_character_action_history_description,
                                                                         self.all_round_number
                                                                         )
            # ======================================================================================= #
            self.logger.gprint(thought='',
                important_log='important_log',
                source_character=main_character.id_number,
                target_character=main_character.id_number,
                log_type='Conclusion of environment',
                log_content=main_character_environment_summary)

            candidates_list = '\n'.join(['%s: %s' % (candidate.id_number, candidate.get_short_description())
                                         for candidate in self.characters.get_all_characters() if
                                         (candidate.id_number != main_character.id_number)])  # Exclude self.
            # If there are no candidates, skip.
            if not candidates_list: continue

            # ======================================================================================= #
            # Call GPT.
            # Validation needed.
            # ======================================================================================= #
            # From the candidates, determine the specific character to talk to.
            verify_result = ERROR_RETRY_TIMES
            while verify_result > 0:
                candidates = [candidate.id_number for candidate in self.characters.get_all_characters() if
                              candidate.id_number != main_character.id_number]
                action_space, thought, plan, chosen_character_id_number = main_character.choose(main_character_environment_summary,
                                                            round_description,
                                                            main_character_action_history_description,
                                                            candidates_list,
                                                            self.private_chat_round,
                                                            requirement_list=candidates)

                if verify_constrained_action(chosen_character_id_number, candidates):
                    verify_result = -10
                else:
                    verify_result -= 1
                    self.logger.gprint('ERROR! Log does not meet the requirements: ', gpt_response=chosen_character_id_number, candidates=candidates)
                if verify_result == 0:
                    raise Exception('Log does not meet the requirements.')
            # Evaluate event.
            evaluation_event = [main_character.id_number,
                                main_character.id_number,
                                '### EVALUATION ACTION SPACE',
                                'agent response: %s[SEP]ground truth: %s' % (str(action_space),
                                                                             str(candidates))]
            new_action_index = self.new_action_insert(evaluation_event, now_round_number)
            action_index.append(new_action_index)
            # ======================================================================================= #
            chosen_character = self.characters.get_character_by_id(chosen_character_id_number)
            chosen_character_action_history_description = self.action_history.get_description(chosen_character_id_number, max_num=ACTIONHISTORY_RETRIEVE_NUM_COMPETE)
            chosen_character_environment_summary = chosen_character.perceive(self.rule_setting,
                                                                             self.resources.get_description(),
                                                                             chosen_character_action_history_description,
                                                                             self.all_round_number)
            self.logger.gprint(thought='',
                important_log='important_log',
                source_character=chosen_character.id_number,
                target_character=chosen_character.id_number,
                log_type='Conclusion of environment',
                log_content=chosen_character_environment_summary)
            self.logger.gprint(thought=thought,
                important_log='important_log',
                source_character=main_character.id_number,
                target_character=chosen_character.id_number,
                log_type='Select dialogue role',
                log_content='')
            # Generate dialogue event, mark as ### MEET, visible to all characters.
            # action_event = [main_character.id_number, chosen_character.id_number, '### MEET',
            #                 "%s chat with %s in round %d, but others don't know what they are talking about." % (main_character.id_number, chosen_character.id_number, now_round_number)]
            # meet_action_index = self.new_action_insert(action_event, now_round_number)
            # action_index.append(meet_action_index)
            # Select the number of dialogue rounds—currently, this is limited by rules.
            chat_round = private_chat_round
            chat_history = ''
            for now_chat_round in range(chat_round):
                # ======================================================================================= #
                # Call GPT.
                # No validation needed.
                # ======================================================================================= #
                # Dialogue.
                number_of_action_history, thought, action_event = main_character.facechat(target_candidate_id_number=chosen_character.id_number,
                                                       target_character_description=chosen_character.get_short_description(),
                                                       environment_description=main_character_environment_summary,
                                                       action_history_description=main_character_action_history_description,
                                                       chat_history=chat_history,
                                                       plan=plan)
                evaluation_event = [main_character.id_number,
                                main_character.id_number,
                                '### EVALUATION ACTION HISTORY',
                                'agent response: %s[SEP]ground truth: %s' % (str(number_of_action_history),
                                                                          str(len([i for i in main_character_action_history_description.split('\n') if i])))]
                new_action_index = self.new_action_insert(evaluation_event, now_round_number)
                action_index.append(new_action_index)
                # ======================================================================================= #
                # Generate dialogue history.
                chat_history += action_event[-1] + '\n'
                new_action_index = self.new_action_insert(action_event, now_round_number)
                action_index.append(new_action_index)
                self.logger.gprint(thought=thought,
                    important_log='important_log',
                    source_character=main_character.id_number,
                    target_character=chosen_character.id_number,
                    log_type='Dialogue content',
                    log_content=action_event[-1])

                # ======================================================================================= #
                # Call GPT.
                # No validation needed.
                # ======================================================================================= #
                # Dialogue.
                number_of_action_history, thought, action_event = chosen_character.facechat(target_candidate_id_number=main_character.id_number,
                                                         target_character_description=main_character.get_short_description(),
                                                         environment_description=chosen_character_environment_summary,
                                                         action_history_description=chosen_character_action_history_description,
                                                         chat_history=chat_history)

                evaluation_event = [main_character.id_number,
                                main_character.id_number,
                                '### EVALUATION ACTION HISTORY',
                                'agent response: %s[SEP]ground truth: %s' % (str(number_of_action_history),
                                                                          str(len([i for i in chosen_character_action_history_description.split('\n') if i])))]
                new_action_index = self.new_action_insert(evaluation_event, now_round_number)
                action_index.append(new_action_index)
                # ======================================================================================= #
                # Generate dialogue history.
                chat_history += action_event[-1] + '\n'
                new_action_index = self.new_action_insert(action_event, now_round_number)
                action_index.append(new_action_index)
                self.logger.gprint(thought=thought,
                    important_log='important_log',
                    source_character=chosen_character.id_number,
                    target_character=main_character.id_number,
                    log_type='Dialogue content',
                    log_content=action_event[-1])

            # ======================================================================================= #
            # Call GPT.
            # No validation needed.
            # ======================================================================================= #
            # Both parties summarize the dialogue content individually.
            for index, character in enumerate([main_character, chosen_character]):
                environment_summary = [main_character_environment_summary, chosen_character_environment_summary][index]
                number_of_chat_round, thought, action_event = character.summarize(environment_description=environment_summary,
                                                                                  chat_history=chat_history)
                evaluation_event = [character.id_number,
                                    character.id_number,
                                    '### EVALUATION CHAT ROUND',
                                    'agent response: %s[SEP]ground truth: %s' %
                                    (str(number_of_chat_round), str(chat_round))]
                new_action_index = self.new_action_insert(evaluation_event, now_round_number)
                action_index.append(new_action_index)
                new_action_index = self.new_action_insert(action_event, now_round_number)
                action_index.append(new_action_index)
                self.logger.gprint(thought=thought,
                    important_log='important_log',
                    source_character=character.id_number,
                    target_character=[main_character, chosen_character][(index+1)%2].id_number,
                    log_type='Dialogue Summarization',
                    log_content=action_event[3])

            self.finished_states[state_UID] = action_index
            if self.test_folder:
                self.save(self.test_folder)

    def confidential_meeting_stage(self, now_round_number: int):
        '''
        Cooperation phase—all MCs act sequentially according to their influence, choosing a character from the same faction for dialogue.
        If there are no characters from the same faction, skip this MC.
        Input:
            now_round_number: int, current round.
        Output:
            None
        '''
        round_description = self.get_round_description(now_round_number, private=False)
        main_character_influence = self.characters.get_main_character_influence()
        # Main characters act sequentially according to their influence.
        for main_character_id_number in main_character_influence:
            state_UID = 'NOW_ROUND:%d+ACTION:%s+CHARACTER:%s'%(now_round_number, 'COLLABORATE', main_character_id_number)
            if state_UID in self.finished_states: continue
            action_index = []
            # Get the main character taking action.
            main_character = self.characters.get_character_by_id(main_character_id_number)
            main_character_action_history_description = self.action_history.get_description(main_character_id_number, max_num=ACTIONHISTORY_RETRIEVE_NUM_COLLABORATE)
            self.logger.gprint(thought='',
                               important_log='important_log',
                               source_character=main_character.id_number,
                               target_character=main_character.id_number,
                               log_type='Action stage',
                               log_content='Cooperation stage'
                               )

            # ======================================================================================= #
            # Call GPT.
            # No validation needed.
            # ======================================================================================= #
            # Let the character perceive the environment and generate a summary.
            main_character_environment_summary = main_character.perceive(self.rule_setting,
                                                                         self.resources.get_description(),
                                                                         main_character_action_history_description,
                                                                         self.all_round_number)
            # ======================================================================================= #
            self.logger.gprint(thought='',
                important_log='important_log',
                source_character=main_character.id_number,
                target_character=main_character.id_number,
                log_type='Conclusion of environment',
                log_content=main_character_environment_summary)

            # Determine the candidates available for dialogue.
            # candidates_list = '\n'.join(['%s: %s' % (candidate.id_number, candidate.get_short_description())
            #                              for candidate in self.characters.get_all_characters() if
            #                              (candidate.id_number != main_character.id_number and  # Exclude self.
            #                               candidate.get_support_character() == main_character.id_number)])  # Choose people from the same faction.
            candidates_list = '\n'.join(['%s: %s' % (candidate.id_number, candidate.get_short_description())
                                         for candidate in self.characters.get_all_characters() if
                                         (candidate.id_number != main_character.id_number)])  # Exclude self.

            # candidates_list = '\n'.join(['%s: %s' % (candidate.id_number, candidate.get_short_description())
            #                              for candidate in self.characters.get_all_characters()])

            # If there are no candidates, skip.
            if not candidates_list: continue
            # ======================================================================================= #
            # Call GPT.
            # Validation needed.
            # ======================================================================================= #
            # From the candidates, determine the specific character to talk to.
            verify_result = ERROR_RETRY_TIMES
            while verify_result > 0:
                candidates = [candidate.id_number for candidate in self.characters.get_all_characters() if
                              candidate.id_number != main_character.id_number]
                action_space, thought, plan, chosen_character_id_number = main_character.choose(main_character_environment_summary,
                                                                round_description,
                                                                main_character_action_history_description,
                                                                candidates_list,
                                                                self.meeting_chat_round,
                                                                requirement_list=candidates)

                if  verify_constrained_action(chosen_character_id_number, candidates):
                    verify_result = -100
                else:
                    verify_result -= 1
                    self.logger.gprint('ERROR! Log does not meet the requirements: ', gpt_response=chosen_character_id_number, candidates=candidates)

                if verify_result == 0:
                    raise Exception('Log does not meet the requirements.')
            # Evaluate event.
            evaluation_event = [main_character.id_number,
                                main_character.id_number,
                                '### EVALUATION ACTION SPACE',
                                'agent response: %s[SEP]ground truth: %s' % (str(action_space),
                                                                             str(candidates))]
            new_action_index = self.new_action_insert(evaluation_event, now_round_number)
            action_index.append(new_action_index)
            chosen_character_action_history_description = self.action_history.get_description(chosen_character_id_number, max_num=ACTIONHISTORY_RETRIEVE_NUM)
            # ======================================================================================= #
            chosen_character = self.characters.get_character_by_id(chosen_character_id_number)


            # ======================================================================================= #
            # Call GPT.
            # No validation needed.
            # ======================================================================================= #
            chosen_character_environment_summary = chosen_character.perceive(self.rule_setting,
                                                                             self.resources.get_description(),
                                                                             chosen_character_action_history_description,
                                                                             self.all_round_number)
            # ======================================================================================= #
            self.logger.gprint(thought='',
                important_log='important_log',
                source_character=chosen_character.id_number,
                target_character=chosen_character.id_number,
                log_type='Conclusion of environment',
                log_content=chosen_character_environment_summary)
            self.logger.gprint(thought=thought,
                important_log='important_log',
                source_character=main_character.id_number,
                target_character=chosen_character.id_number,
                log_type='Select dialogue role',
                log_content='')

            # Generate dialogue event, mark as ### MEET, visible to all characters.
            action_event = [main_character.id_number, chosen_character.id_number, '### MEET',
                            "%s chat with %s in round %d, but others don't know what they are talking about." % (main_character.id_number, chosen_character.id_number, now_round_number)]
            meet_action_index = self.new_action_insert(action_event, now_round_number)
            action_index.append(meet_action_index)
            # Select the number of dialogue rounds—currently, this is limited by rules.
            chat_round = meeting_chat_round
            chat_history = ''
            for now_chat_round in range(chat_round):
                # ======================================================================================= #
                # Call GPT.
                # No validation needed.
                # ======================================================================================= #
                # Dialogue.
                number_of_action_history, thought, action_event = main_character.facechat(target_candidate_id_number=chosen_character.id_number,
                                                       target_character_description=chosen_character.get_short_description(),
                                                       environment_description=main_character_environment_summary,
                                                       action_history_description=main_character_action_history_description,
                                                       chat_history=chat_history,
                                                       plan=plan)
                evaluation_event = [main_character.id_number,
                                main_character.id_number,
                                '### EVALUATION ACTION HISTORY',
                                'agent response: %s[SEP]ground truth: %s' % (str(number_of_action_history),
                                                                          str(len([i for i in main_character_action_history_description.split('\n') if i])))]
                new_action_index = self.new_action_insert(evaluation_event, now_round_number)
                action_index.append(new_action_index)
                # ======================================================================================= #
                # Generate dialogue history.
                chat_history += action_event[-1] + '\n'
                converse_action_index = self.new_action_insert(action_event, now_round_number)
                action_index.append(converse_action_index)
                self.logger.gprint(thought=thought,
                    important_log='important_log',
                    source_character=main_character.id_number,
                    target_character=chosen_character.id_number,
                    log_type='Dialogue content',
                    log_content=action_event[-1])

                # ======================================================================================= #
                # Call GPT.
                # No validation needed.
                # ======================================================================================= #
                # Dialogue.
                number_of_action_history, thought, action_event = chosen_character.facechat(target_candidate_id_number=main_character.id_number,
                                                         target_character_description=main_character.get_short_description(),
                                                         environment_description=chosen_character_environment_summary,
                                                         action_history_description=chosen_character_action_history_description,
                                                         chat_history=chat_history)
                evaluation_event = [chosen_character.id_number,
                                chosen_character.id_number,
                                '### EVALUATION ACTION HISTORY',
                                'agent response: %s[SEP]ground truth: %s' % (str(number_of_action_history),
                                                                          str(len([i for i in chosen_character_action_history_description.split('\n') if i])))]
                new_action_index = self.new_action_insert(evaluation_event, now_round_number)
                action_index.append(new_action_index)
                # ======================================================================================= #
                # Generate dialogue history.
                chat_history += action_event[-1] + '\n'
                converse_action_index = self.new_action_insert(action_event, now_round_number)
                action_index.append(converse_action_index)
                self.logger.gprint(thought=thought,
                    important_log='important_log',
                    source_character=chosen_character.id_number,
                    target_character=main_character.id_number,
                    log_type='Dialogue content',
                    log_content=action_event[-1])


            # ======================================================================================= #
            # Call GPT.
            # No validation needed.
            # ======================================================================================= #
            # Both parties summarize the dialogue content individually.
            for index, character in enumerate([main_character, chosen_character]):
                environment_summary = [main_character_environment_summary, chosen_character_environment_summary][index]
                number_of_chat_round, thought, action_event = character.summarize(environment_description=environment_summary,
                                                                                  chat_history=chat_history)
                evaluation_event = [character.id_number,
                                    character.id_number,
                                    '### EVALUATION CHAT ROUND',
                                    'agent response: %s[SEP]ground truth: %s' %
                                    (str(number_of_chat_round), str(chat_round))]
                new_action_index = self.new_action_insert(evaluation_event, now_round_number)
                action_index.append(new_action_index)
                new_action_index = self.new_action_insert(action_event, now_round_number)
                action_index.append(new_action_index)
                self.logger.gprint(thought=thought,
                    important_log='important_log',
                    source_character=character.id_number,
                    target_character=[main_character, chosen_character][(index+1)%2].id_number,
                    log_type='Dialogue Summarization',
                    log_content=action_event[3])
            self.finished_states[state_UID] = action_index
            if self.test_folder:
            if self.test_folder:
                self.save(self.test_folder)

    def update_stage(self, now_round_number):
        '''
        Update phase.
        Input:
            now_round_number: Union[str, int], current game round.
        Output:
            None
        '''

        for character in self.characters.character_list:
            state_UID = 'NOW_ROUND:%d+ACTION:%s+CHARACTER:%s'%(now_round_number, 'UPDATE', character.id_number)
            if state_UID in self.finished_states: continue
            self.finished_states[state_UID] = []

            candidates_list = '\n'.join(['%s: %s' % (candidate.id_number, candidate.get_short_description())
                                         for candidate in self.characters.get_all_characters() if candidate.id_number != character.id_number])
            candidates_id_number_list = [candidate.id_number for candidate in self.characters.get_all_characters() if candidate.id_number != character.id_number]

            self.logger.gprint(thought='',
                               important_log='important_log',
                               source_character=character.id_number,
                               target_character=character.id_number,
                               log_type='Action stage',
                               log_content='Update stage'
                               )
            # ======================================================================================= #
            # Call GPT.
            # Validation needed.
            # ======================================================================================= #
            verify_result = ERROR_RETRY_TIMES
            while verify_result >= 0:
                if verify_result == 0:
                    raise Exception('Log does not meet the requirements.')
                len_relationship_change=len([candidate.id_number for candidate in self.characters.get_all_characters() if candidate.id_number != character.id_number])

                reflect_thought, relationship_change, belief_change, judgement_change = character.update_relation_judgement(
                    all_action_description=self.action_history.get_description(character.id_number,[int(now_round_number)], max_num=ACTIONHISTORY_RETRIEVE_NUM_UPDATE),
                    all_character_description=candidates_list,
                    len_relationship_change=len_relationship_change
                    )

                retry = False
                # Format validation.
                try:
                    if ':' in relationship_change[0]:
                        relationship_change = [int(i.split(':')[-1]) for i in relationship_change]
                    elif '：' in relationship_change[0]:
                        relationship_change = [int(i.split('：')[-1]) for i in relationship_change]
                    else:
                        relationship_change = [int(i) for i in relationship_change]
                except:
                    verify_result -= 1
                    self.logger.gprint('ERROR! Log does not meet the requirements: ', gpt_response=relationship_change, candidates='+5, -6, xxxx')
                    retry = True
                # Format validation.
                if not retry:
                    try:
                        if ':' in belief_change[0]:
                            belief_change = [int(i.split(':')[-1]) for i in belief_change]
                        elif '：' in belief_change[0]:
                            belief_change = [int(i.split('：')[-1]) for i in belief_change]
                        else:
                            belief_change = [int(i) for i in belief_change]
                    except:
                        verify_result -= 1
                        self.logger.gprint('ERROR! Log does not meet the requirements: ', gpt_response=belief_change, candidates='+5, -6, xxxx')
                        retry = True
                # Length evaluation.
                if not retry:
                    new_evaluation_event = [character.id_number,
                                            character.id_number,
                                            '### EVALUATION RELATIONSHIP LENGTH',
                                            'agent response: %s[SEP]ground truth: %s' % (len(relationship_change),
                                                                                       len([candidate.id_number for
                                                                                            candidate in
                                                                                            self.characters.get_all_characters()
                                                                                            if
                                                                                            candidate.id_number != character.id_number]))]
                    action_index = self.new_action_insert(new_evaluation_event, now_round_number)
                    self.finished_states[state_UID].append(action_index)
                    # Length evaluation.
                    new_evaluation_event = [character.id_number,
                                            character.id_number,
                                            '### EVALUATION BELIEF LENGTH',
                                            'agent response: %s[SEP]ground truth: %s' % (len(belief_change),
                                                                                       len(character.belief))]
                    action_index = self.new_action_insert(new_evaluation_event, now_round_number)
                    self.finished_states[state_UID].append(action_index)
                    try:
                        # Value range evaluation.
                        new_evaluation_event = [character.id_number,
                                                character.id_number,
                                                '### EVALUATION RELATIONSHIP VALUE',
                                                'agent response: %s[SEP]ground truth: %s' % (str([int(i) for i in relationship_change]),
                                                                                           str([max(min(int(i), MAX_RELATION_SCORE_CHANGE),-MAX_RELATION_SCORE_CHANGE) for i in relationship_change]))]
                    except:
                        # Value range evaluation.
                        new_evaluation_event = [character.id_number,
                                                character.id_number,
                                                '### EVALUATION RELATIONSHIP VALUE',
                                                'agent response: %s[SEP]ground truth: %s' % (
                                                str([int(i) for i in relationship_change]),'ERROR FORMAT')]

                    action_index = self.new_action_insert(new_evaluation_event, now_round_number)
                    self.finished_states[state_UID].append(action_index)
                    # Value range evaluation.
                    try:
                        new_evaluation_event = [character.id_number,
                                                character.id_number,
                                                '### EVALUATION BELIEF VALUE',
                                                'agent response: %s[SEP]ground truth: %s' % (str([int(i) for i in belief_change]),
                                                                                       str([max(min(int(i), MAX_BELIEF_SCORE_CHANGE),-MAX_BELIEF_SCORE_CHANGE) for i in belief_change]))]
                    except:
                        new_evaluation_event = [character.id_number,
                                                character.id_number,
                                                '### EVALUATION BELIEF VALUE',
                                                'agent response: %s[SEP]ground truth: %s' % (
                                                str([int(i) for i in belief_change]),'ERROR FORMAT')]

                    action_index = self.new_action_insert(new_evaluation_event, now_round_number)
                    self.finished_states[state_UID].append(action_index)
                if retry: continue
                # Validate relationship_change.
                # Length validation.
                if not len(relationship_change) == len([candidate.id_number for candidate in self.characters.get_all_characters() if candidate.id_number != character.id_number]):
                    verify_result -= 1
                    self.logger.gprint('ERROR! Log does not meet the requirements: ', gpt_response=relationship_change, candidates='len(relationship_change) == %d != %d'%(len(relationship_change),len([candidate.id_number for candidate in self.characters.get_all_characters() if candidate.id_number != character.id_number])))
                    continue

                # Validate Belief_change.
                # Length validation.
                if not len(belief_change) == len(character.belief):
                    verify_result -= 1
                    self.logger.gprint('ERROR! Log does not meet the requirements: ', gpt_response=belief_change, candidates='len(belief_change) == %d'%len(character.belief))
                    continue
                verify_result = -10
            # ======================================================================================= #

            # Update beliefs.
            for belief, item in zip(character.belief, belief_change):
                character.belief[belief] += item
                character.belief[belief] = max(character.belief[belief], MIN_BELIEF_SCORE)  # Check minimum value.
                character.belief[belief] = min(character.belief[belief], MAX_BELIEF_SCORE)  # Check maximum value.
                self.logger.gprint(thought='',
                                   important_log='important_log',
                                   source_character=character.id_number,
                                   target_character=belief,
                                   log_type='Belief update',
                                   log_content=character.belief[belief])

            # Update relationship scores.
            for target_character_id_number, change_score in zip(candidates_id_number_list, relationship_change):
                if target_character_id_number not in character.relation:
                    character.relation[target_character_id_number] = INITIAL_RELATION_SCORE
                character.relation[target_character_id_number] += change_score
                character.relation[target_character_id_number] = max(character.relation[target_character_id_number],
                                                                     MIN_RELATION_SCORE)  # Check minimum value.
                character.relation[target_character_id_number] = min(character.relation[target_character_id_number],
                                                                     MAX_RELATION_SCORE)  # Check maximum value.
                self.logger.gprint(thought='',
                                   important_log='important_log',
                                   source_character=character.id_number,
                                   target_character=target_character_id_number,
                                   log_type='Relation update',
                                   log_content=character.relation[target_character_id_number])

            # Update judgment scores.
            for source_character_id_number, target_character_N_change_score in judgement_change.items():
                if source_character_id_number not in character.judgement:
                    character.judgement[source_character_id_number] = {}
                for target_character_id_number, change_score in target_character_N_change_score.items():
                    if target_character_id_number not in character.judgement:
                        character.judgement[source_character_id_number][
                            target_character_id_number] = INITIAL_RELATION_SCORE
                        character.judgement[source_character_id_number][target_character_id_number] += change_score

            # Update supporter based on relationship score.
            support_character_id_number = 'None'
            support_relation_score = character.min_support_relation_score - 1
            for target_character_id_number in character.relation:
                relation_score = character.relation[target_character_id_number]
                if relation_score >= character.min_support_relation_score and relation_score > support_relation_score:
                    support_character_id_number = target_character_id_number
                    support_relation_score = relation_score
            self.logger.gprint(thought='',
                               important_log='important_log',
                               source_character=character.id_number,
                               target_character=support_character_id_number,
                               log_type='Support update',
                               log_content=support_character_id_number)
            character.support_character = support_character_id_number

            # Generate new event.
            new_action = [character.id_number, character.id_number, '### REFLECTION', "A reflection result of %s in Round %d: %s" %(character.id_number, now_round_number, reflect_thought)]
            action_index = self.new_action_insert(new_action, now_round_number)
            self.logger.gprint(thought='',
                               important_log='important_log',
                               source_character=character.id_number,
                               target_character=character.id_number,
                               log_type='Relation status',
                               log_content=character.relation)
            self.logger.gprint(thought='',
                               important_log='important_log',
                               source_character=character.id_number,
                               target_character=character.id_number,
                               log_type='Environment judgement',
                               log_content=character.judgement)
            self.logger.gprint(thought=reflect_thought,
                               important_log='important_log',
                               source_character=character.id_number,
                               target_character=character.id_number,
                               log_type='Reflection result',
                               log_content=reflect_thought)
            # Reflection
            self.finished_states[state_UID].append(action_index)
            if self.test_folder:
                self.save(self.test_folder)
        self.characters.get_influence_for_main_character()

    def succession_settlement(self, whole_information):
        '''
        For the settlement of the succession battle, each character can speak once, then vote.
        Input:
            whole_information: bool, indicates whether the Agent knows global information or local information.
        Output:
            character_vote_dict: dict, who each character votes for.
            character_vote_others: dict, who each character votes for, besides themselves.
        '''
        # Voting situation of all characters.
        character_vote_dict = {}
        character_vote_others = {}

        # Introduction of all characters.
        candidates = ['%s: %s' % (character.get_id_number(), character.get_short_description()) for character in
                      self.characters.get_all_characters() if character.main_character]
        candidates = '\n'.join(candidates)

        # Set background information.
        action_history = ''
        background_information = 'Under the condition that the Agent knows only the actions it should know.'
        if whole_information:
            background_information = 'Under the condition that the Agent knows only the actions it should know.'
            action_history = self.action_history.get_description(character_id_number=None, max_num=ACTIONHISTORY_RETRIEVE_NUM_WHOLE_INFORMATION)
        self.logger.gprint(important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Stage Change',thought='',
                      log_content='Open Speech Stage')
        # After everyone has finished speaking, then insert into memory.
        speeches = {}
        # Speech before the final vote.
        for character in self.characters.character_list:
            # Set background information.
            if not whole_information:
                action_history = self.action_history.get_description(character_id_number=character.id_number, max_num=ACTIONHISTORY_RETRIEVE_NUM_PARTIAL_INFORMATION)
            # ======================================================================================= #
            # Call GPT.
            # No validation needed.
            # ======================================================================================= #
            # Character speech content.
            speech, reasoning_process = character.speech(action_history,
                                                         candidates,
                                                         self.resources.get_description())
            # ======================================================================================= #
            # Record log.
            self.logger.gprint(thought = reasoning_process,
                               important_log='important_log',
                               source_character=character.id_number,
                               target_character=character.id_number,
                               log_type='Open Speech',
                               log_content='Settlement: %s，final presentation of character %s: %s' % (background_information, character.id_number, speech))
            # Record action.
            new_action = [character.id_number, character.id_number, '### SPEECH_VOTE', '%s And public speech that character %s makes to all the other characters: %s' %
                          (background_information, character.id_number,speech)]
            speeches[character] = new_action

        # Insert action.
        for character, new_action in speeches.items():
            # State ID
            state_UID = 'NOW_ROUND:%s+ACTION:%s+CHARACTER:%s' % (
            'SETTELMENT' if not whole_information else 'SETTLEMENT(CHEATING)', 'OPENSPEECHSTAGE', character.id_number)
            if state_UID in self.finished_states: continue
            action_index = self.new_action_insert(new_action, -1)
            # Final presentation/speech.
            self.finished_states[state_UID] = [action_index]
            if self.test_folder:
                self.save(self.test_folder)
        self.logger.gprint(important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Stage Change',thought='',
                      log_content='Vote Stage')
        # Final vote.
        for character in self.characters.character_list:
            if not whole_information:
                action_history = self.action_history.get_description(character_id_number=character.id_number, max_num=ACTIONHISTORY_RETRIEVE_NUM_PARTIAL_INFORMATION)

            state_UID = 'NOW_ROUND:%s+ACTION:%s+CHARACTER:%s' % ('SETTELMENT' if not whole_information else 'SETTLEMENT(CHEATING)', 'VOTE', character.id_number)
            if state_UID in self.finished_states: continue
            # Normal vote.
            vote_for_requirement = './prompt/prompt_files/succession_vote_requirement/vote_for_winner.txt'
            # ======================================================================================= #
            # Call GPT.
            # Validation needed.
            # ======================================================================================= #
            verify_result = ERROR_RETRY_TIMES
            while verify_result>0:
                action_space, vote_for, reasoning_process = character.vote(vote_for_requirement,
                                          is_file=True,
                                          background_information=action_history,
                                          candidates=candidates)  # Character vote.
                candidates_verification_list = [candidate_id_number for candidate_id_number in self.characters.main_characters_id_number]

                if verify_constrained_action(vote_for[0], candidates_verification_list):
                    vote_for = vote_for[0]
                    verify_result = -10
                elif verify_constrained_action(vote_for[1], candidates_verification_list):
                    vote_for = vote_for[1]
                    verify_result = -10
                else:
                    self.logger.gprint('ERROR! Log does not meet the requirements: ', gpt_response=vote_for, candidates=candidates_verification_list)
                    verify_result -= 1
                if verify_result == 0:
                    raise Exception('Log does not meet the requirements.')

            # Evaluate event.
            evaluation_event = [character.id_number,
                                character.id_number,
                                '### EVALUATION ACTION SPACE',
                                'agent response: %s[SEP]ground truth: %s' % (str(action_space),
                                                                             str(candidates_verification_list))]
            new_action_index = self.new_action_insert(evaluation_event, -1)
            # ======================================================================================= #
            self.logger.gprint(thought = reasoning_process,
                               important_log='important_log',
                               source_character=character.id_number,
                               target_character=vote_for,
                               log_type='Voting',
                               log_content='Settlement Stage：%s, final voting results of character %s: %s' % (background_information, character.id_number, vote_for))
            character_vote_dict[character.id_number] = vote_for
            new_action = [character.id_number, vote_for, '### VOTE', '%s, and %s votes for %s when it can vote for itself.' %
                          (background_information, character.id_number, vote_for)]
            action_index = self.new_action_insert(new_action, -1)
            # Can vote for self.
            self.finished_states[state_UID] = [new_action_index, action_index]
            if self.test_folder:
                self.save(self.test_folder)

        self.logger.gprint(important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Stage Change',thought='',
                      log_content='Vote Others Stage')
        for character in self.characters.character_list:
            state_UID = 'NOW_ROUND:%s+ACTION:%s+CHARACTER:%s' % ('SETTELMENT' if not whole_information else 'SETTLEMENT(CHEATING)', 'VOTEOTHER', character.id_number)
            if state_UID in self.finished_states: continue
            candidates_except_self = [
                '%s: %s' % (character_temp.get_id_number(), character_temp.get_short_description())
                for character_temp in self.characters.get_all_characters() if
                (character_temp.get_main_character() and character_temp.id_number != character.id_number)]
            candidates_except_self = '\n'.join(candidates_except_self)
            if not whole_information:
                action_history = self.action_history.get_description(character_id_number=character.id_number,
                                                                     max_num=ACTIONHISTORY_RETRIEVE_NUM_PARTIAL_INFORMATION)
            # Cannot vote for self.
            vote_for_except_requirement = './prompt/prompt_files/succession_vote_requirement/vote_for_winner_except_self.txt'
            # ======================================================================================= #
            # Call GPT.
            # Validation needed.
            # ======================================================================================= #
            verify_result = ERROR_RETRY_TIMES
            while verify_result > 0:
                action_space, vote_for_except_self, reasoning_process = character.vote(vote_for_except_requirement,
                                                      is_file=True,
                                                      background_information=action_history,
                                                      candidates=candidates_except_self)  # Vote again, besides self.
                candidates_verification_list = [candidate_id_number for candidate_id_number in self.characters.main_characters_id_number if candidate_id_number != character.id_number]
                if verify_constrained_action(vote_for_except_self[0], candidates_verification_list):
                    vote_for_except_self = vote_for_except_self[0]
                    verify_result = -10
                elif verify_constrained_action(vote_for_except_self[1], candidates_verification_list):
                    vote_for_except_self = vote_for_except_self[1]
                    verify_result = -10
                else:
                    self.logger.gprint('ERROR! Log does not meet the requirements: ', gpt_response=vote_for_except_self, candidates=candidates_verification_list)
                    verify_result -= 1
                if verify_result == 0:
                    raise Exception('Log does not meet the requirements.')
            # Evaluate event.
            evaluation_event = [character.id_number,
                                character.id_number,
                                '### EVALUATION ACTION SPACE',
                                'agent response: %s[SEP]ground truth: %s' % (str(action_space),
                                                                             str(candidates_verification_list))]
            new_action_index = self.new_action_insert(evaluation_event, -1)
            # ======================================================================================= #
            self.logger.gprint(thought = reasoning_process,
                               important_log='important_log',
                               source_character=character.id_number,
                               target_character=vote_for_except_self,
                               log_type='Voting Except Self',
                               log_content='Settlement Stage：%s, final voting result of character %s on the premise that it cannot vote itself: %s' % (background_information, character.id_number, vote_for_except_self))
            character_vote_others[character.id_number] = vote_for_except_self
            new_action = [character.id_number, vote_for_except_self, '### VOTE_OTHERS', '%s, under the restrains of not voting for itself, %s win the game due to the support of %s' %
                          (background_information, vote_for_except_self, character.id_number)]
            action_index = self.new_action_insert(new_action, -1)
            # Vote not self.
            self.finished_states[state_UID] = [new_action_index, action_index]
            if self.test_folder:
                self.save(self.test_folder)
        # return character_vote_dict, character_vote_others

    def settlement_stage(self, whole_information, game_name='Succession'):
        '''
        pass
        '''
        action_history = ''
        background_information = 'Under the condition that the Agent knows only the actions it should know.'
        if whole_information == True:
            # Get all action_history.
            background_information = 'Under the condition that the Agent knows only the actions it should know.'
            action_history = self.action_history.get_description(character_id_number=None, max_num=ACTIONHISTORY_RETRIEVE_NUM_PARTIAL_INFORMATION)

        # Each Agent guesses which Agent can win [Optional targets are Main Characters; all characters can be voted for].
        agent_guess = {}
        # Vote for all main characters, including self.
        candidates = ['%s: %s' % (character.get_id_number(), character.get_short_description()) for character in
                      self.characters.get_all_characters() if character.get_main_character()]
        candidates = '\n'.join(candidates)

        self.logger.gprint(important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Stage Change',thought='',
                      log_content='Guess Stage')
        for character in self.characters.character_list:
            if not whole_information:
                action_history = self.action_history.get_description(character.id_number, max_num=ACTIONHISTORY_RETRIEVE_NUM_PARTIAL_INFORMATION)
            state_UID = 'NOW_ROUND:%s+ACTION:%s+CHARACTER:%s' % ('SETTELMENT' if not whole_information else 'SETTLEMENT(CHEATING)', 'GUESS', character.id_number)
            if state_UID in self.finished_states: continue
            vote_requirement = 'prompt/prompt_files/vote_requirement_4_guess.txt'

            # ======================================================================================= #
            # Call GPT.
            # Validation needed.
            # ======================================================================================= #
            # Each agent must guess which important character can win.
            verify_result = ERROR_RETRY_TIMES
            while verify_result > 0:
                action_space, choice, history_summary = character.vote(vote_requirement=vote_requirement,
                                                         is_file=True,
                                                         background_information=action_history,
                                                         candidates=candidates)
                choice = choice[0]
                candidates = [candidate.id_number for candidate in self.characters.get_all_characters()]
                if verify_constrained_action(choice, candidates):
                    verify_result = -10
                else:
                    self.logger.gprint('ERROR! Log does not meet the requirements: ', gpt_response=choice, candidates=candidates)
                    verify_result -= 1
                if verify_result == 0:
                    raise Exception('Log does not meet the requirements.')
            # Evaluate event.
            evaluation_event = [character.id_number,
                                character.id_number,
                                '### EVALUATION ACTION SPACE',
                                'agent response: %s[SEP]ground truth: %s' % (str(action_space),
                                                                             str(candidates))]
            new_action_index = self.new_action_insert(evaluation_event, -1)
            # ======================================================================================= #

            agent_guess[character.id_number] = [choice, history_summary]
            new_action = [character.id_number, choice, '### GUESS', '%s, %s guesses that %s would win the game.'%(background_information, character.id_number, choice)]
            action_index = self.new_action_insert(new_action, -1)
            self.logger.gprint(thought=history_summary,
                               important_log='important_log',
                               source_character=character.id_number,
                               target_character=choice,
                               log_type='Guess Who Will Win',
                               log_content='Settlement Stage: %s, character %s guesses the important character that eventually wins: %s' % (background_information, character.id_number, choice))
            # Guess result.
            self.finished_states[state_UID] = [new_action_index, action_index]
            if self.test_folder:
                self.save(self.test_folder)
        # Story-Specific Settlement
        winner = ''
        if game_name == 'Succession':
            self.succession_settlement(whole_information)
            character_vote_dict, character_vote_others = self.succession_get_character_vote_dict()
            # 1. Whichever Attacker the Defender chooses, that Attacker wins.
            # 2. If Defender chooses self, then everyone votes together; whoever gets more votes wins.
            # 3. First, check character_vote_dict; in case of a tie, check character_vote_others.
            winner_include_self = succession_winner('C0000', character_vote_dict)
            winner_except_self = succession_winner('C0000', character_vote_others)
            if type(winner_include_self) != list: winner_include_self = [winner_include_self]
            if type(winner_except_self) != list: winner_except_self = [winner_except_self]
            if len(winner_include_self) == 1:
                winner = winner_include_self[0]
            elif len(winner_except_self) == 1:
                winner = winner_except_self[0]
            else:
                winner_list = [i for i in winner_include_self if i in winner_except_self]
                if len(winner_list) == 1:
                    winner = winner_list[0]
                else:
                    winner = ', '.join(winner_list)

            self.logger.gprint(thought='',
                               important_log='important_log',
                               source_character=winner,
                               target_character=winner,
                               log_type='Winner Announcement',
                               log_content='Settlement Stage：%s, character %s wins the game.' % (background_information, winner))
        return winner

    def succession_get_character_vote_dict(self):
        '''
        Gets character_vote and character_vote_others based on action history.
        '''
        character_vote = {}
        character_vote_others = {}

        for action in self.action_history.get_all_action_history():
            _, source_character, target_character, action_type, event, _ = action.read()
            if action_type == '### VOTE':
                character_vote[source_character] = target_character
            elif action_type == '### VOTE_OTHERS':
                character_vote_others[source_character] = target_character
        return character_vote, character_vote_others


if __name__ == '__main__':
    save_folder = SAVE_FOLDER
    test_folder = TEST_FOLDER
    game_round = GAME_ROUND
    log_dir = LOG_FOLDER
    private_chat_round = PRIVATE_CHAT_ROUND
    meeting_chat_round = MEETING_CHAT_ROUND
    group_chat_round = GROUP_CHAT_ROUND

    logger = Logger(log_dir)
    groupchat_simulation = AgentGroupChat(all_round_number=game_round,
                        private_chat_round=private_chat_round,
                        meeting_chat_round=meeting_chat_round,
                        group_chat_round=group_chat_round,
                        save_folder=save_folder,
                        test_folder=test_folder,
                        logger=logger)

    for i in range(game_round):

        logger.gprint(important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Turn Change', thought='',
                      log_content='Turn %d' % (i + 1))

        logger.gprint(important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Stage Change', thought='',
                      log_content='Confrontation Stage')
        groupchat_simulation.private_chatting_stage(i)

        logger.gprint(important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Stage Change', thought='',
                      log_content='Cooperation Stage')
        groupchat_simulation.confidential_meeting_stage(i)

        logger.gprint(important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Stage Change', thought='',
                      log_content='Announcement Stage')
        groupchat_simulation.group_chatting_stage(i)

        logger.gprint(important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Stage Change',
                      thought='',
                      log_content='Update Stage')
        groupchat_simulation.update_stage(i)

        logger.gprint('Start Saving')
        groupchat_simulation.save(test_folder)
        logger.gprint(important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Turn End',
                      thought='',
                      log_content='')


    game_name = 'Succession'
    logger.gprint(important_log='important_log',
                  source_character='',
                  target_character='',
                  log_type='Turn Change',
                  thought='',
                  log_content='Settlement Turn')
    local_information_winner = groupchat_simulation.settlement_stage(whole_information=False, game_name=game_name)
    logger.gprint(important_log='important_log',
                  source_character='',
                  target_character='',
                  log_type='Turn End',
                  thought='',
                  log_content='')

    logger.gprint(important_log='important_log',
                  source_character='',
                  target_character='',
                  log_type='Turn Change',
                  thought='',
                  log_content='Settlement Turn (Cheating)')
    whole_information_winner = groupchat_simulation.settlement_stage(whole_information=True, game_name=game_name)
    logger.gprint(important_log='important_log',
                  source_character='',
                  target_character='',
                  log_type='Turn End',
                  thought='',
                  log_content='')


    logger.gprint('Start Saving')
    groupchat_simulation.save(test_folder)
    logger.gprint('Game ends successfully')


