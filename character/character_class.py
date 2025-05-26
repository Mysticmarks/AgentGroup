import json
import os
import random
from collections import defaultdict
import re # For parsing "To [ID]:"
from config import * # type: ignore
from config import DEFAULT_GGUF_MODEL_NAME # Import the default GGUF model name

from logger_class import Logger
from typing import List, Optional, Any, Tuple, Dict # Ensure List, Optional, Tuple, Dict are imported
from dataclasses import dataclass # For PlaceholderMessage
from collections import deque # For recent_conversation_summary
from prompt.gpt_structure import generate_with_response_parser # For LLM calls

from character.action_modules.summarization import run_summarization
from character.action_modules.choose import run_choose
from character.action_modules.facechat import run_facechat
from character.action_modules.perceive import run_perceive
from character.action_modules.reflection import run_reflect
from character.action_modules.vote import run_vote
from character.action_modules.groupchat import run_groupchat, run_speech


@dataclass
class PlaceholderMessage: 
    speaker_id: str
    content: str
    timestamp: float 
    addressed_to_id: Optional[str] = None


class Character:
    def __init__(self, id_number=None, main_character=False, save_file_folder=None, logger: Logger = None) -> None: # type: ignore
        if logger:
            self.logger = logger
        else:
            self.logger = None # type: ignore
        self.main_character = main_character
        self.type = "ai"  # Default type
        self.influence = 0
        self.support_character = ''  # Stores id_number
        self.objective = ''
        self.name = ''
        self.scratch = ''
        self.background = ''
        self.engine = ''
        self.belief = {}
        self.judgement = defaultdict(lambda: INITIAL_RELATION_SCORE) # type: ignore
        self.relation = defaultdict(lambda: INITIAL_RELATION_SCORE) # type: ignore
        self.id_number = ''
        if id_number:
            self.id_number = id_number
        
        self.recent_conversation_summary: deque = deque(maxlen=10) 

        if save_file_folder:
            self.load(save_file_folder)

    def give_influence(self, influence: int) -> None:
        self.influence = influence

    def get_influence(self) -> int:
        return self.influence

    def get_objective(self) -> str:
        return self.objective

    def get_support_character(self) -> str:
        if self.support_character:
            return self.support_character
        else:
            return 'Nobody, you support yourself for the moment.'
    def get_id_number(self) -> str:
        return self.id_number

    def get_main_character(self) -> bool:
        return self.main_character

    def get_relationship(self) -> dict:
        return self.relation

    def summarize(self, environment_description, chat_history):
        number_of_action_history, thought, new_action_history = run_summarization(self.id_number,
                                                                             self.get_self_description(),
                                                                             environment_description,
                                                                             chat_history,
                                                                             engine=self.engine,
                                                                             logger=self.logger)
        return number_of_action_history, thought, new_action_history

    def load(self, save_file_folder) -> None:
        if not save_file_folder.endswith('.json'):
            save_file_folder = os.path.join(save_file_folder, self.id_number + '.json')

        save_file = save_file_folder
        json_data = json.load(open(save_file, encoding='utf-8'))

        self.name = json_data['name']
        if not self.id_number: # If id_number was not provided to __init__
            self.id_number = json_data['id_name']
        self.main_character = True if json_data['main_character'] == 'True' else False
        self.support_character = json_data['support_character']
        self.objective = json_data['objective']
        self.scratch = json_data['scratch']
        self.background = json_data['background']
        self.engine = json_data.get('engine') 

        if self.engine is None or self.engine == "" or self.engine == "default_gguf":
            if DEFAULT_GGUF_MODEL_NAME and DEFAULT_GGUF_MODEL_NAME.strip():
                if self.logger:
                    self.logger.gprint(f"Character {self.id_number if self.id_number else json_data.get('id_name', 'Unknown')}: Engine not specified or set to default. Using DEFAULT_GGUF_MODEL_NAME: '{DEFAULT_GGUF_MODEL_NAME}'. Original engine value: '{json_data.get('engine')}'", level="INFO")
                self.engine = DEFAULT_GGUF_MODEL_NAME
            elif self.logger:
                 self.logger.gprint(f"Character {self.id_number if self.id_number else json_data.get('id_name', 'Unknown')}: Engine not specified or set to default, but DEFAULT_GGUF_MODEL_NAME is not set in config. Engine remains: '{self.engine}'.", level="WARNING")
        
        if self.engine is None:
            self.engine = ""

        self.belief = json_data['belief']
        self.judgement = json_data['judgement']
        self.relation = json_data['relation']
        self.portrait = json_data.get('portrait', '') 
        self.small_portrait = json_data.get('small_portrait', '') 
        self.type = json_data.get('type', 'ai') 
        self.min_support_relation_score = MIN_SUPPORT_RELATION_SCORE # type: ignore

    def save(self, save_file_folder) -> None:
        if not save_file_folder.endswith('.json'):
            if not os.path.exists(save_file_folder):
                os.makedirs(save_file_folder)
            save_file_folder = os.path.join(save_file_folder, self.id_number + '.json')
        save_file = save_file_folder
        json_data = {'name': self.name,
                     'id_name': self.id_number,
                     'main_character': 'True' if self.main_character else 'False',
                     'support_character': self.support_character,
                     'objective': self.objective,
                     'scratch': self.scratch,
                     'background': self.background,
                     'engine': self.engine,
                     'belief': self.belief,
                     'judgement': self.judgement,
                     'portrait': self.portrait,
                     'small_portrait': self.small_portrait,
                      'relation': self.relation,
                      'type': self.type} 
        open(save_file, 'w', encoding='utf-8').write(json.dumps(json_data, ensure_ascii=False, indent=4))

    def get_self_description(self) -> str:
        description = 'You: %s.\n' % self.id_number
        description += 'Your goal: %s\n' % self.objective
        if not self.main_character:
            if self.support_character:
                description += 'You are currently supporting %s in achieving his/her goals.\n' % (
                    self.support_character)
            else:
                description += 'You are not supporting anyone at the moment.\n'
        description += 'Here is your role setting: %s\n' % self.scratch
        description += 'In the public eye, you are: %s\n' % self.background
        if self.main_character:
            description += 'Your thought: %s' % self.get_main_belif()
        return description.strip()

    def get_main_belif(self) -> str:
        main_belief = []
        main_belief_score = 0
        for belief, score in self.belief.items():
            if score > main_belief_score:
                main_belief_score = score
                main_belief = []
            if score == main_belief_score:
                main_belief.append(belief.strip('。'))
        return '; '.join(main_belief) + '. '

    def get_all_belief(self) -> str:
        all_belief = ''
        for belief in self.belief:
            all_belief += belief+' : '+str(self.belief[belief]) + '\n'
        return all_belief

    def get_all_belief_number(self) -> int:
        return len(self.belief)

    def get_short_description(self) -> str:
        description = self.background
        return description

    def update_relation_judgement(self, all_action_description: str,
                                  all_character_description: str,
                                  len_relationship_change: int):
        change_case = [random.randint(-10, 10) for i in range(100)]
        change_case = ['+%d' % i if i > 0 else str(i) for i in change_case]
        case_of_relationship_change = ', '.join(change_case[:int(len_relationship_change)])
        case_of_belief_change = ', '.join(change_case[:self.get_all_belief_number()])
        reflect_thought, relationship_change, belief_change, judgement_change = run_reflect(self.id_number,
                                                                                           self.get_self_description(),
                                                                                           self.get_all_belief(),
                                                                                           all_action_description,
                                                                                           all_character_description,
                                                                                           str(len_relationship_change),
                                                                                           str(self.get_all_belief_number()),
                                                                                           case_of_relationship_change,
                                                                                           case_of_belief_change,
                                                                                           engine=self.engine,
                                                                                           logger=self.logger)
        return reflect_thought, relationship_change, belief_change, judgement_change

    def speech(self, action_history_desc, candidates, resources):
        speech, reasoning_process = run_speech(self.id_number,
                                               self.get_self_description(),
                                               action_history_desc,
                                               candidates,
                                               resources,
                                               self.get_support_character(),
                                               engine=self.engine,
                                               logger=self.logger
                                               )
        return speech, reasoning_process

    def groupchat(self, action_history_desc, candidates, resources, round_description):
        speech, reasoning_process = run_groupchat(self.id_number,
                                               self.get_self_description(),
                                               action_history_description=action_history_desc,
                                               candidates_description=candidates,
                                               resources=resources,
                                               round_description=round_description,
                                               support_character=self.get_support_character(),
                                               engine=self.engine,
                                               logger=self.logger
                                               )
        return speech, reasoning_process

    def choose(self,
            environment_summary: str,
            round_description: str,
            action_history_description: str,
            candidates_description: str,
            chat_round_num: int,
            requirement_list: list=None):
        action_history, thought, plan, candidate = run_choose(self.id_number,
                                                           self.get_self_description(),
                                                           environment_summary,
                                                           round_description,
                                                           action_history_description,
                                                           candidates_description,
                                                           str(chat_round_num),
                                                           engine=self.engine,
                                                           requirement_list=requirement_list,
                                                           logger=self.logger)
        return action_history, thought, plan, candidate

    def get_belief_and_score(self):
        belief_score_dict = {}
        for belief, score in self.belief.items():
            belief_score_dict[belief] = score
        return belief_score_dict

    def vote(self,
             vote_requirement,
             is_file: bool,
             background_information: str,
             candidates: str,
             requirement_list: list=None):
        if is_file:
            vote_requirement = \
            open(vote_requirement, encoding='utf-8').read().split('<commentblockmarker>###</commentblockmarker>')[
                -1].strip()
        action_space, choice, reasoning_process = run_vote(self.id_number,
                                                           self.get_self_description(),
                                                           '\n'.join(['%s: %s' % (belief, score) for belief, score in
                                                                      self.get_belief_and_score().items()]),
                                                           vote_requirement,
                                                           background_information,
                                                           candidates,
                                                           self.get_support_character(),
                                                           engine=self.engine,
                                                           requirement_list=requirement_list,
                                                           logger=self.logger)
        return action_space, choice, reasoning_process

    def facechat(self,
                 target_candidate_id_number: str,
                 target_character_description: str,
                 environment_description: str,
                 action_history_description: str,
                 chat_history: str,
                 plan: str = None):
        if not plan:
            plan = 'You do not have a specific plan; please reply based on your character settings and objectives.'
        number_of_action_history, thought, new_action_history = run_facechat(self.id_number,
                                                                             target_candidate_id_number,
                                                                             self.get_self_description(),
                                                                             target_character_description,
                                                                             environment_description,
                                                                             action_history_description,
                                                                             chat_history,
                                                                             plan,
                                                                             engine=self.engine,
                                                                             logger=self.logger)
        return number_of_action_history, thought, new_action_history

    def perceive(self, rule_setting: str, all_resource_description: str, action_history_description: str, chat_round_number) -> str:
        environment_description = run_perceive(self.id_number,
                                               self.get_self_description(),
                                               rule_setting,
                                               all_resource_description,
                                               action_history_description,
                                               chat_round_number=chat_round_number,
                                               support_character=self.get_support_character(),
                                               engine=self.engine,
                                               logger=self.logger)
        return environment_description

    # --- Methods for Natural Conversation Model Integration ---

    def perceive_conversation_update(self, new_message: PlaceholderMessage, full_history: Optional[List[PlaceholderMessage]] = None) -> None:
        formatted_message = f"[{new_message.speaker_id}]: {new_message.content}"
        if new_message.addressed_to_id: # Add addressing info to summary if present
            formatted_message += f" (to {new_message.addressed_to_id})"
        self.recent_conversation_summary.append(formatted_message)
        
        if self.logger:
            log_content = f"Perceived message from {new_message.speaker_id}: '{new_message.content}'"
            if new_message.addressed_to_id == self.id_number:
                log_content += " (addressed to me)"
            elif new_message.addressed_to_id:
                 log_content += f" (addressed to {new_message.addressed_to_id})"
            self.logger.gprint(
                thought=f"Updating recent conversation summary with: {formatted_message}",
                important_log='debug', 
                source_character=self.id_number,
                target_character=new_message.speaker_id,
                log_type='ConversationPerception',
                log_content=log_content
            )

    def decide_and_generate_response(self, 
                                     full_history: Optional[List[PlaceholderMessage]] = None, 
                                     participant_list: Optional[List[Dict[str, str]]] = None
                                     ) -> Optional[Tuple[str, Optional[str]]]:
        if self.type == "human": 
            if self.logger:
                self.logger.gprint(f"Character {self.id_number} is human, skipping LLM response generation.", level="INFO", log_type="ResponseDecision")
            return None

        is_directly_addressed = False
        last_message_speaker = None
        if full_history and full_history[-1].addressed_to_id == self.id_number:
            is_directly_addressed = True
            last_message_speaker = full_history[-1].speaker_id
        
        if not self.recent_conversation_summary and not is_directly_addressed:
            if self.logger:
                self.logger.gprint(f"Character {self.id_number}: No recent messages and not directly addressed. Skipping response.", level="DEBUG", log_type="ResponseDecision")
            return None

        formatted_recent_history = "\n".join(self.recent_conversation_summary)
        
        formatted_participant_list_str = "Current participants in the conversation:\n"
        if participant_list:
            for p_info in participant_list:
                if p_info['id_name'] == self.id_number: # Don't list the agent itself
                    continue
                formatted_participant_list_str += f"- {p_info['id_name']} ({p_info.get('name', 'Unknown Name')})\n"
        else:
            formatted_participant_list_str += "- (Participant list not available)\n"

        addressing_info_for_prompt = ""
        if is_directly_addressed:
            addressing_info_for_prompt = f"You were directly addressed by {last_message_speaker}. Consider addressing your response to them."
        else:
            addressing_info_for_prompt = "You can address a specific participant by starting your response with 'To [participant_id]: Message' or address the group generally. Example: 'To C0001: Hello there!'"


        prompt = (
            f"You are {self.name} ({self.id_number}).\n"
            f"{self.get_self_description()}\n\n"
            f"{formatted_participant_list_str}\n" # Added participant list here
            f"Recent conversation history (most recent last):\n{formatted_recent_history}\n\n"
            f"---\n"
            f"Given your personality, objectives, the participants, and the conversation history:\n"
            f"1. Should you respond to this? (Answer strictly with \"Yes\" or \"No\").\n"
            f"2. If Yes, what is your response? {addressing_info_for_prompt} If you choose not to respond (answered \"No\" to question 1), just write \"No response needed.\" for question 2."
        )

        if self.logger:
            self.logger.gprint(
                thought="Constructed prompt for LLM based on self-description, participants, and recent conversation for selective response.",
                important_log='debug',
                source_character=self.id_number,
                log_type='PromptConstruction',
                log_content=f"Prompt for {self.id_number}:\n{prompt}"
            )

        try:
            raw_llm_response = generate_with_response_parser(
                message_or_prompt=prompt,
                engine=self.engine, 
                logger=self.logger,
                func_name="decide_and_generate_response_selective"
            )

            if not raw_llm_response:
                if self.logger: self.logger.gprint(f"{self.id_number} LLM returned empty or None. No response.", "DEBUG")
                return None

            lines = raw_llm_response.strip().split('\n')
            if not lines:
                if self.logger: self.logger.gprint(f"{self.id_number} LLM response split into empty lines. No response.", "DEBUG")
                return None

            decision_line = lines[0].lower()
            should_respond = "yes" in decision_line
            
            if self.logger: self.logger.gprint(f"{self.id_number} LLM decision to respond: {'Yes' if should_respond else 'No'}. Decision line: '{lines[0]}'", "DEBUG")

            if not should_respond:
                return None

            response_content_from_llm = ""
            if len(lines) > 1:
                response_started = False
                temp_response_lines = []
                for i in range(1, len(lines)):
                    line_strip = lines[i].strip()
                    if line_strip.startswith("2.") or line_strip.lower().startswith("if yes, what is your response?"):
                        response_started = True 
                        question_marker_end = line_strip.lower().find("if yes, what is your response?") + len("if yes, what is your response?")
                        if question_marker_end < len(line_strip): 
                            content_part = line_strip[question_marker_end:].strip()
                            if content_part and content_part.lower() != "no response needed.":
                                temp_response_lines.append(content_part)
                        continue 
                    if response_started or not line_strip.startswith("1.") : 
                        if line_strip.lower() == "no response needed.": 
                             if self.logger: self.logger.gprint(f"{self.id_number} LLM said Yes then 'No response needed.'", "DEBUG")
                             return None
                        temp_response_lines.append(lines[i]) 
                response_content_from_llm = "\n".join(temp_response_lines).strip()
            else: 
                 if self.logger: self.logger.gprint(f"{self.id_number} LLM said Yes but provided no further lines for response.", "DEBUG")
                 return None


            if not response_content_from_llm or response_content_from_llm.lower() == "no response needed.":
                if self.logger: self.logger.gprint(f"{self.id_number} Extracted response is effectively 'No response needed' or empty. Silent.", "DEBUG")
                return None
            if response_content_from_llm == "...": 
                if self.logger: self.logger.gprint(f"{self.id_number} LLM response is '...'. Silent.", "DEBUG")
                return None
                
            parsed_addressed_to_id: Optional[str] = None
            actual_response_content = response_content_from_llm 

            match = re.match(r"To\s+([^\s:]+)\s*:(.*)", response_content_from_llm, re.IGNORECASE | re.DOTALL)
            if match:
                potential_id = match.group(1).strip()
                content_after_to = match.group(2).strip()
                
                if potential_id and content_after_to: 
                    parsed_addressed_to_id = potential_id
                    actual_response_content = content_after_to
                    if self.logger:
                        self.logger.gprint(f"Agent {self.id_number} addressed response to '{parsed_addressed_to_id}'. Content: '{actual_response_content}'", level="DEBUG")
            
            if is_directly_addressed and parsed_addressed_to_id is None:
                parsed_addressed_to_id = last_message_speaker
                if self.logger:
                     self.logger.gprint(f"Agent {self.id_number} was directly addressed by {last_message_speaker} and did not use 'To:', defaulting response to {last_message_speaker}.", level="DEBUG")

            if self.logger:
                log_msg = f"{self.id_number} responds: '{actual_response_content}'"
                if parsed_addressed_to_id:
                    log_msg += f" (to {parsed_addressed_to_id})"
                self.logger.gprint(
                    thought="LLM generated a response after selective decision.",
                    important_log='info',
                    source_character=self.id_number,
                    log_type='ResponseGenerated',
                    log_content=log_msg
                )
            return actual_response_content, parsed_addressed_to_id

        except Exception as e:
            if self.logger:
                self.logger.gprint(
                    thought=f"Error during LLM call or structured response parsing for {self.id_number}: {e}",
                    important_log='error',
                    source_character=self.id_number,
                    log_type='LLMError',
                    log_content=str(e)
                )
            return None
