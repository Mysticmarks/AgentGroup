import time
import collections
from typing import List, Dict, Optional, Any # Any might still be used for human_cli_user object
import os 
import time 
import collections 
from typing import List, Dict, Optional, Any 
from dataclasses import dataclass, asdict 
import json 
import asyncio 

from character.character_class import Character, PlaceholderMessage 
from character.all_character_class import AllCharacter 
from logger_class import Logger 
from config import LOG_FOLDER, SAVE_FOLDER, INITIAL_RELATION_SCORE 
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

# Constants for conversation history persistence
CONVERSATION_LOG_DIR = "./conversation_logs/"
DEFAULT_CONVERSATION_FILE = os.path.join(CONVERSATION_LOG_DIR, "current_chat.json")


@dataclass
class CoreMessage: 
    """Represents a single message in the conversation within ConversationManager."""
    speaker_id: str
    content: str
    timestamp: float
    addressed_to_id: Optional[str] = None

ConversationHistory = List[CoreMessage]

class ConversationManager:
    """Manages the flow of conversation, participants, and message history."""

    def __init__(self, 
                 logger_instance: Optional[Logger] = None): 
        """Initializes the ConversationManager."""
        self.participants: Dict[str, Character] = {}
        self.message_history: ConversationHistory = []
        self.message_queue = collections.deque() 
        self.logger = logger_instance if logger_instance else Logger(LOG_FOLDER, log_level="DEBUG") 
        self.running = True 
        self.websocket_broadcast_callback: Optional[Any] = None # Renamed from socket_broadcaster

        os.makedirs(CONVERSATION_LOG_DIR, exist_ok=True)
        self._load_conversation_history()

    def set_websocket_broadcast_callback(self, callback_fn: Any):
        """Sets the callback function for broadcasting messages via WebSocket."""
        self.websocket_broadcast_callback = callback_fn
        self.logger.gprint("WebSocket broadcast callback has been set.", level="INFO")

    def _save_conversation_history(self):
        """Saves the current conversation history to a JSON file."""
        try:
            history_to_save = [asdict(msg) for msg in self.message_history]
            with open(DEFAULT_CONVERSATION_FILE, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, indent=2, ensure_ascii=False)
            self.logger.gprint(f"Conversation history saved to {DEFAULT_CONVERSATION_FILE}", level="DEBUG")
        except Exception as e:
            self.logger.gprint(f"Error saving conversation history: {e}", level="ERROR")

    def _load_conversation_history(self):
        """Loads conversation history from a JSON file if it exists."""
        if os.path.exists(DEFAULT_CONVERSATION_FILE):
            try:
                with open(DEFAULT_CONVERSATION_FILE, 'r', encoding='utf-8') as f:
                    loaded_messages_data = json.load(f)
                    self.message_history = [CoreMessage(**msg_data) for msg_data in loaded_messages_data]
                self.logger.gprint(f"Loaded {len(self.message_history)} messages from {DEFAULT_CONVERSATION_FILE}", level="INFO")
            except Exception as e:
                self.logger.gprint(f"Error loading conversation history from {DEFAULT_CONVERSATION_FILE}: {e}. Starting with empty history.", level="ERROR")
                self.message_history = [] 
        else:
            self.logger.gprint("No existing conversation history found. Starting fresh.", level="INFO")
            self.message_history = []


    def add_participant(self, participant_id: str, participant_obj: Character):
        """Adds a Character participant to the conversation."""
        if participant_id in self.participants:
            self.logger.gprint(f"Participant with ID '{participant_id}' already exists. Overwriting.", level="WARNING")
        self.participants[participant_id] = participant_obj
        self.logger.gprint(f"Participant {participant_id} ({participant_obj.name if hasattr(participant_obj, 'name') else 'Unknown Name'}) added.", level="INFO")

    def remove_participant(self, participant_id: str):
        """Removes a participant from the conversation."""
        if participant_id in self.participants:
            del self.participants[participant_id]
            print(f"Participant {participant_id} removed.")
        else:
            self.logger.gprint(f"Attempted to remove non-existent participant with ID '{participant_id}'.", level="WARNING")

    def submit_message(self, speaker_id: str, content: str, addressed_to_id: Optional[str] = None):
        """Creates a CoreMessage object and adds it to the message queue."""
        if speaker_id != "human_cli_user" and speaker_id not in self.participants:
             self.logger.gprint(f"Message submitted by unknown or non-character participant ID '{speaker_id}'.", level="WARNING")

        message = CoreMessage( 
            speaker_id=speaker_id,
            content=content,
            timestamp=time.time(),
            addressed_to_id=addressed_to_id
        )
        self.message_queue.append(message)
        log_msg_detail = f" (to {addressed_to_id})" if addressed_to_id else ""
        self.logger.gprint(f"Message from '{speaker_id}' queued: \"{content}\"{log_msg_detail}", level="DEBUG")


    def _notify_participants(self, message: CoreMessage):
        """Notifies all other participants about a new message by calling their perceive_conversation_update."""
        for participant_id, participant_char in self.participants.items():
            if participant_char and participant_id != message.speaker_id: 
                try:
                    participant_char.perceive_conversation_update(new_message=message, full_history=self.message_history)
                except Exception as e:
                    self.logger.gprint(f"Error notifying participant {participant_id} about message from {message.speaker_id}: {e}", level="ERROR")


    def process_message_queue(self):
        """Processes messages from the queue, adds to history, notifies participants, and triggers AI responses."""
        processed_this_cycle = 0
        messages_to_process_this_turn = list(self.message_queue)
        self.message_queue.clear() 

        for message in messages_to_process_this_turn:
            self.message_history.append(message)
            self._save_conversation_history() 
            processed_this_cycle += 1
            
            timestamp_str = time.strftime('%H:%M:%S', time.localtime(message.timestamp))
            if message.addressed_to_id:
                formatted_msg_for_display = f"[{timestamp_str}] {message.speaker_id} (to {message.addressed_to_id}): {message.content}"
            else:
                formatted_msg_for_display = f"[{timestamp_str}] {message.speaker_id}: {message.content}"
            
            # Always print to console for CLI/local visibility
            print(formatted_msg_for_display) 
            
            # If a WebSocket broadcast callback is set, use it
            if self.websocket_broadcast_callback:
                try:
                    # The callback (e.g., socket_manager.broadcast) is async.
                    # asyncio.create_task is appropriate here as process_message_queue
                    # might be called from a synchronous context (like the original __main__ loop)
                    # or an async context (like the server's background loop).
                    asyncio.create_task(self.websocket_broadcast_callback(formatted_msg_for_display))
                except Exception as e:
                    if self.logger: # Check if logger is available
                        self.logger.gprint(f"Error invoking websocket_broadcast_callback: {e}", level="ERROR")
            
            self._notify_participants(message)

        if processed_this_cycle > 0:
            active_participants = list(self.participants.items()) 
            for participant_id, participant_char in active_participants:
                if participant_char and participant_char.type == "ai":
                    try:
                        response_data = participant_char.decide_and_generate_response(full_history=self.message_history)
                        if response_data:
                            response_content, response_addressed_to_id = response_data
                            if response_content: 
                                self.logger.gprint(f"AI '{participant_id}' generated response: \"{response_content}\" (to: {response_addressed_to_id})", level="INFO")
                                self.submit_message(
                                    speaker_id=participant_id, 
                                    content=response_content, 
                                    addressed_to_id=response_addressed_to_id
                                )
                    except Exception as e:
                         self.logger.gprint(f"Error during AI response generation for {participant_id}: {e}", level="ERROR", error=True) # type: ignore
        

    def create_and_add_agent(self,
                             id_name: str,
                             name: str,
                             objective: str,
                             scratch: str,
                             background: str,
                             engine: str,
                             character_type: str = "ai",
                             is_main_character: bool = False,
                             portrait_path: Optional[str] = None,
                             small_portrait_path: Optional[str] = None,
                             beliefs: Optional[Dict] = None, 
                             relations: Optional[Dict] = None, 
                             judgements: Optional[Dict] = None):
        """Creates a new AI agent, saves its configuration, and adds it to the conversation."""
        self.logger.gprint(f"Attempting to create and add agent: ID '{id_name}', Name '{name}'", level="INFO")

        if id_name in self.participants:
            self.logger.gprint(f"Error: Agent with ID '{id_name}' already exists in participants. Creation aborted.", level="ERROR")
            return None

        new_agent = Character(id_number=id_name, logger=self.logger)
        new_agent.name = name
        new_agent.objective = objective
        new_agent.scratch = scratch
        new_agent.background = background
        new_agent.engine = engine
        new_agent.type = character_type  
        new_agent.main_character = is_main_character
        
        new_agent.portrait = portrait_path if portrait_path else './storage/images/portrait/character/default.jpg'
        new_agent.small_portrait = small_portrait_path if small_portrait_path else './storage/images/small_portrait/character/default.jpg'
        
        if beliefs is not None:
            new_agent.belief = beliefs
        elif not hasattr(new_agent, 'belief') or new_agent.belief is None : new_agent.belief = {}
        
        if relations is not None: 
            for key, value in relations.items(): new_agent.relation[key] = value
        elif not hasattr(new_agent, 'relation') or new_agent.relation is None: new_agent.relation = collections.defaultdict(lambda: INITIAL_RELATION_SCORE) # type: ignore

        if judgements is not None: 
            for key, value in judgements.items(): new_agent.judgement[key] = value
        elif not hasattr(new_agent, 'judgement') or new_agent.judgement is None: new_agent.judgement = collections.defaultdict(lambda: INITIAL_RELATION_SCORE) # type: ignore


        user_agents_dir = "./user_profiles/"
        try:
            os.makedirs(user_agents_dir, exist_ok=True)
            new_agent.save(user_agents_dir) 
            self.logger.gprint(f"Agent '{name}' (ID: {id_name}) configuration saved to {os.path.join(user_agents_dir, id_name + '.json')}", level="INFO")
        except Exception as e:
            self.logger.gprint(f"Error saving agent configuration for '{id_name}': {e}", level="ERROR")
            return None 

        self.add_participant(new_agent.id_number, new_agent)
        self.logger.gprint(f"Agent '{name}' (ID: {id_name}) successfully created, saved, and added to the conversation.", level="INFO")
        return new_agent

    def run_main_loop(self, cli_user_id: Optional[str] = None):
        """Runs a simple command-line interface loop for message submission."""
        history_file = os.path.join(os.path.expanduser("~"), ".natural_conv_history")
        session = PromptSession(history=FileHistory(history_file))
        
        prompt_speaker_display = cli_user_id if cli_user_id else "You" 
        
        self.logger.gprint(f"Conversation Manager CLI started. Input as '{prompt_speaker_display}'. Type '/quit' to exit or '/participants' to list.", level="INFO")

        if not self.participants:
            self.logger.gprint("Warning: No participants (AI or Human) have been added. Conversation will be empty until participants are added.", level="WARNING")
        
        if cli_user_id and cli_user_id not in self.participants:
            self.logger.gprint(f"Warning: Specified CLI user ID '{cli_user_id}' not found among participants. Input will be attributed to a generic 'human_cli_user'.", level="WARNING")
        elif not cli_user_id:
             self.logger.gprint(f"Warning: No specific CLI user ID provided. Input will be attributed to a generic 'human_cli_user'.", level="WARNING")

        try:
            while self.running:
                self.process_message_queue()
                if cli_user_id: 
                    prompt_text = f"{prompt_speaker_display}: "
                    try:
                        user_input = session.prompt(prompt_text, auto_suggest=AutoSuggestFromHistory())
                        
                        cleaned_input = user_input.strip()
                        if cleaned_input.lower() == '/quit':
                            self.logger.gprint("'/quit' command received. Exiting CLI loop.", level="INFO")
                            self.running = False
                            break 
                        elif cleaned_input.lower() == '/participants':
                            print("Participants:", list(self.participants.keys()))
                            continue
                        
                        if cleaned_input: 
                            speaker_for_submission = cli_user_id 
                            self.submit_message(speaker_id=speaker_for_submission, content=cleaned_input)
                    except EOFError: 
                        self.logger.gprint("EOF received (Ctrl+D). Exiting CLI loop.", level="INFO")
                        self.running = False
                        break
                    except KeyboardInterrupt: 
                        self.logger.gprint("\nKeyboard interrupt received. Exiting CLI loop.", level="INFO")
                        self.running = False
                        break
                else: 
                    if not self.message_queue and not any(p.type == 'ai' for p in self.participants.values()):
                        self.logger.gprint("No human CLI user and no AI messages pending. Loop will be idle. Consider adding participants or providing a CLI user.", level="INFO")
                    pass 
                time.sleep(0.1) 
                
        except Exception as e: 
            self.logger.gprint(f"Unexpected error in main loop: {e}", level="ERROR")
        finally:
            self._save_conversation_history() 
            self.logger.gprint("Conversation Manager CLI stopped.", level="INFO")


if __name__ == "__main__":
    try:
        main_logger = Logger(LOG_FOLDER, log_level="INFO") 
        main_logger.gprint("Main logger initialized for natural_conversation_core.py direct run.", level="INFO")
    except Exception as e:
        print(f"Critical Error: Failed to initialize main logger: {e}. Exiting.")
        exit() 

    manager = ConversationManager(logger_instance=main_logger)

    scenario_path = SAVE_FOLDER 
    main_logger.gprint(f"Loading characters from scenario: {scenario_path} and user_profiles/", level="INFO")
    
    try:
        all_participants_loader = AllCharacter(save_folder=scenario_path, logger=main_logger)
    except Exception as e:
        main_logger.gprint(f"Error initializing AllCharacter: {e}. Check config.py SAVE_FOLDER and user_profiles directory.", level="CRITICAL")
        exit() 

    if not all_participants_loader.get_all_characters():
        main_logger.gprint("No characters (AI or Human) were loaded. Check scenario path and user_profiles. Exiting.", level="WARNING")

    for character_obj in all_participants_loader.get_all_characters():
        main_logger.gprint(f"Adding participant to ConversationManager: {character_obj.id_number} (Name: {character_obj.name}, Type: {character_obj.type})", level="INFO")
        manager.add_participant(character_obj.id_number, character_obj)

    active_human_id: Optional[str] = None
    for char_id, char_obj in manager.participants.items():
        if char_obj.type == "human":
            active_human_id = char_id
            main_logger.gprint(f"Active CLI user set to: {active_human_id} (Name: {char_obj.name})", level="INFO")
            break 
    
    if active_human_id is None:
        main_logger.gprint("No human participant found in loaded characters. CLI input will be attributed to 'human_cli_user' (default).", level="WARNING")

    manager.run_main_loop(cli_user_id=active_human_id)

    main_logger.gprint("Attempting to dynamically create and add 'user_ai_carl'...", level="INFO")
    carl = manager.create_and_add_agent(
        id_name="user_ai_carl",
        name="Carl",
        objective="To observe and learn about human conversations.",
        scratch="Carl is a newly instantiated AI, eager to understand interactions.",
        background="Carl is an AI observer.",
        engine="default_gguf" 
    )
    if carl:
        main_logger.gprint(f"Agent '{carl.name}' (ID: {carl.id_number}) dynamically added to manager. Total participants: {len(manager.participants)}", level="INFO")
    else:
        main_logger.gprint("Failed to dynamically create and add 'user_ai_carl'.", level="ERROR")
    
    # Example of how to make the new agent speak (optional, for further testing)
    # if carl:
    #    manager.submit_message(speaker_id=carl.id_number, content="Hello everyone, I'm Carl, the new AI here!")
    #    manager.process_message_queue() # Process Carl's intro
