from collections import defaultdict
import os
import json


class Action:
    def __init__(self, idd, source_character_id_number, to_character_id_number, action_type, action, happen_time=0):
        '''
        Action Type:
            ### MEET: Information visible to everyone—who met whom.
            ### CHAT: Information visible only to both parties of the dialogue—who talked to whom.
            ### REFLECT: Visible only to oneself—reflection results.

        Input:
            source_character_id_number, str
            to_character_id_number, str
            action_type, str
            action, str
            happen_time, int

        Output:
            None
        '''
        self.build_up(idd, source_character_id_number, to_character_id_number, action_type, action, happen_time)

    def read(self):
        return self.id, self.source_character_id_number, self.to_character_id_number, self.action_type, self.action, self.happen_time

    def build_up(self, idd, source_character_id_number, to_character_id_number, action_type, action, happen_time):
        '''
        Constructs an Action: source_character performed an action of action_type towards to_character, specifically: action, occurring at happen_time.
        Input:
            idd, int
            source_character_id_number, str
            to_character_id_number, str
            action_type, str
            action, str
            happen_time, int

        Output:
            None
        '''
        self.id = idd
        self.source_character_id_number = source_character_id_number
        self.to_character_id_number = to_character_id_number
        self.action_type = action_type
        self.action = action
        self.happen_time = happen_time


class ActionHistory:
    def __init__(self, save_folder=None, basic_setting_file=None)->None:
        '''
        Initializes the ActionHistory class.
        Input:
            save_folder: str, archive location address.
        Output:
            None
        '''
        self.action_history = []
        self.all_happen_time = set()
        if save_folder:
            self.initialize(save_folder, basic_setting_file)

    def initialize(self, save_folder, basic_setting_file)->int:
        '''
        Initializes the entire ActionHistory class from a specific archive.
        Input:
            save_folder: str, archive location address.
        Output:
            success_number: int, how many actions were successfully read.
        '''
        effective_ids = []
        if basic_setting_file:
            basic_setting = json.load(open(basic_setting_file, encoding='utf-8'))
            state_uid = basic_setting['finished_states']
            for i, j in state_uid.items():
                effective_ids.extend(j)
            effective_ids = list(set(effective_ids))
        success_number = 0
        for file in os.listdir(save_folder):
            save_file = os.path.join(save_folder, file)
            success_number += self.load(save_file, effective_ids)
        return success_number

    def save(self, save_folder, action_number_in_each_file=1000)->None:
        '''
        Saves all actions to the specified folder.
        Input:
            save_folder: str, storage address.
            action_number_in_each_file: int, how many actions each file stores.
        Output:
            None
        '''
        start_file_index = 0  # Alternatively, integer division can be used directly.
        start_action_index = 0
        # Guard against an error.
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        fw = open(os.path.join(save_folder, '%04d.json' % start_file_index), 'w', encoding='utf-8')
        while start_action_index < len(self.action_history):
            if start_action_index % action_number_in_each_file == 0:
                fw = open(os.path.join(save_folder, '%04d.json' % start_file_index), 'w', encoding='utf-8')
                start_file_index += 1

            action = self.action_history[start_action_index]
            json_data = {'id':action.id,
                         'source_character_id_number': action.source_character_id_number,
                         'to_character_id_number': action.to_character_id_number,
                         'action_type': action.action_type,
                         'action': action.action,
                         'happen_time': action.happen_time}
            fw.write(json.dumps(json_data, ensure_ascii=False) + '\n')
            start_action_index += 1

    def load(self, save_log_file, effectiveness_ids:list)->int:
        '''
        Reads all actions that occurred in the game from a file.
        Input:
            save_file: str, address of the saved file.
        Output:
            success_number: int, how many historical actions were read.
        '''
        success_number = 0
        json_file = open(save_log_file, encoding='utf-8')
        for json_line in json_file:
            json_data = json.loads(json_line)
            if json_data['id'] not in effectiveness_ids: continue
            action = Action(json_data['id'],
                            json_data['source_character_id_number'],
                            json_data['to_character_id_number'],
                            json_data['action_type'],
                            json_data['action'],
                            json_data['happen_time'])
            self.insert_action(action)
            success_number += 1
        return success_number
    def get_description(self, character_id_number, happen_time_list:list=None, max_num:int=9999, type_list=None)->str:
        '''
        Gets a description of all actions a specific character can see.
        Input:
            character_id_number: str
            happen_time_list: list [int], retrieve events at these specific time points.
            max_num: int, maximum number of memories to retrieve.
        Output:
            action_description: str
        '''
        visible_action = self.retrieve_character_history(character_id_number, happen_time_list, type_list)
        if not visible_action: return 'This is the first round of the game, and nothing has happened so far.'
        action_description = ''

        # Later events are placed towards the end of visible_action.
        for index, action in enumerate(visible_action[-100:][::-1]):
            action_description += action.action.strip()+'\n'
            if index + 1 > max_num: break  # Exceeded maximum number.
        action_description = action_description.strip()
        if action_description == '':
            action_description = 'This is the first round of the game, and nothing has happened so far.'
        return action_description.strip()

    def get_all_action_history(self)->list:
        '''
        Gets the history of all actions for the self character.
        Input:
            None
        Output:
            action_history: list
        '''
        return self.action_history

    def insert_action(self, new_action: Action)->int:
        '''
        Inserts an action.
        Input:
            new_action
        Output:
            None
        '''
        new_action.id = len(self.action_history)
        self.all_happen_time.add(new_action.happen_time)
        self.action_history.append(new_action)
        return new_action.id

    def extend_actions(self, new_action_list: list)->None:
        '''
        Inserts a list of actions.
        Input:
            new_action_list: list (Action)
        Output:
            None
        '''
        for new_action in new_action_list:
            self.insert_action(new_action)

    def retrieve_character_history(self, character_id_number: str, happen_time_list:list=None,type_list:list=None)->list:
        '''
        Gets all action history visible to a specific character.
        Input:
            character_id_number: str
            happen_time_list: list [int], retrieve events at these specific time points.
        Output:
            visible_action: list, all actions visible to the character.
        '''
        if not happen_time_list: happen_time_list = list(self.all_happen_time)
        # if character_id_number:
        #     visible_action = [i for i in self.action_history if self.see_action(i, happen_time_list, character_id_number)]
        # else:

        visible_action = [i for i in self.action_history if self.see_action(i, happen_time_list, character_id_number, type_list)]
        # Chronological order—items at the end of the list are the newest actions.
        visible_action = visible_action[::-1]
        return visible_action

    def see_action(self, action:Action, happen_time_list, character_id_number,type_list):
        if type_list:
            pass
        else:
            if action.action_type in ['### SAY']: return False  # ### SAY is only visible during Summarize, and summarize does not call action_history.
            if action.happen_time not in happen_time_list: return False
            if character_id_number in [action.to_character_id_number, action.source_character_id_number]: return True
            if action.action_type in ['### MEET', '### SPEECH_NORMAL', '### SPEECH_VOTE']: return True  # Everyone's MEET and SPEECH events can be seen.
            if action.action_type not in ['### REFLECT', '### CHAT_SUMMARIZATION']: return False  # Everyone can only recall what they said and what they reflected on.
