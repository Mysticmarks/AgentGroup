from character.character_class import Character
import os
from config import *
from collections import defaultdict


class AllCharacter:
    def __init__(self, save_folder=None, logger=None) -> None:
        self.character_dict = {}
        self.character_list = []

        self.main_characters_id_number = []
        self.npc_characters_id_number = [] # Initialize npc_characters_id_number

        self.initial_relation_score = INITIAL_RELATION_SCORE
        self.relationships = defaultdict(dict)

        self.initial_influence_score = int(INITIAL_INFLUENCE_SCORE)
        self.main_character_influence = defaultdict(lambda: self.initial_influence_score)

        if save_folder: # This is for AI characters
            self.initialize(save_folder, logger=logger, character_type_filter="ai")
        
        # Load human user profiles
        self.load_user_profiles(logger=logger)
        
        self.get_influence_for_main_character()

    def load_user_profiles(self, logger=None, user_profile_folder="./user_profiles/"):
        if not os.path.exists(user_profile_folder):
            if logger:
                logger.gprint(f"User profiles folder '{user_profile_folder}' not found. Skipping loading of human users.", level="WARNING")
            else:
                print(f"Warning: User profiles folder '{user_profile_folder}' not found. Skipping loading of human users.")
            return 0
        
        success_number = 0
        for file in os.listdir(user_profile_folder):
            if file.endswith(".json"):
                user_profile_path = os.path.join(user_profile_folder, file)
                try:
                    # Character __init__ calls load if save_file_folder is provided
                    character = Character(save_file_folder=user_profile_path, logger=logger)
                    # The Character.load method should set self.type based on the JSON content.
                    # If "type" is "human", it will be set. Engine should also be set from JSON.
                    self.append(character) # Add to self.character_dict and self.character_list
                    success_number += 1
                    if logger:
                        logger.gprint(f"Successfully loaded human user profile: {character.id_number} ({character.name}) with type: {character.type} and engine: {character.engine}", level="INFO")
                    else:
                        print(f"Successfully loaded human user profile: {character.id_number} ({character.name}) with type: {character.type} and engine: {character.engine}")

                    # Human users are typically not main characters in the simulation's objective sense,
                    # unless explicitly defined in their JSON with "main_character": "True".
                    # The existing logic for main_characters_id_number in initialize() will handle this
                    # if a human user profile happens to be marked as a main_character.
                    # No special handling for self.main_characters_id_number needed here unless requirements change.

                except Exception as e:
                    if logger:
                        logger.gprint(f"Error loading user profile from {user_profile_path}: {e}", level="ERROR")
                    else:
                        print(f"Error loading user profile from {user_profile_path}: {e}")
        return success_number

    def get_influence_for_main_character(self) -> dict:
        # 重建每个main character的influence score
        for character in self.character_list:
            if character.main_character:
                self.main_character_influence[character.id_number] = int(self.initial_influence_score)

        # 计算每个main character的influence score
        for character in self.character_list:
            if character.main_character: continue
            support_character_id_number = character.support_character
            support_character = self.get_character_by_id(support_character_id_number)
            if support_character.main_character:
                # Only add influence if the supporting character is an AI (or not explicitly human with different rules)
                # And the supported character is a main AI character.
                # This logic might need refinement based on how human influence is intended to work.
                # For now, assuming human players don't have 'influence' stat in the same way AI do,
                # or their influence is handled differently.
                if character.type == "ai" and support_character and support_character.main_character:
                    self.main_character_influence[support_character_id_number] += int(character.get_influence())
        return self.main_character_influence

    def initialize(self, save_folder, logger=None, character_type_filter=None) -> int:
        success_number = 0
        if not os.path.exists(save_folder):
            if logger:
                logger.gprint(f"Save folder '{save_folder}' not found. Skipping loading of AI characters.", level="WARNING")
            else:
                print(f"Warning: Save folder '{save_folder}' not found. Skipping loading of AI characters.")
            return 0
            
        for file in os.listdir(save_folder):
            if not file.endswith(".json"): # Ensure only JSON files are processed
                continue
            character_file = os.path.join(save_folder, file)
            try:
                character = Character(save_file_folder=character_file, logger=logger)
                
                # Apply filter if specified (e.g., only load "ai" characters here)
                if character_type_filter and character.type != character_type_filter:
                    if logger:
                        logger.gprint(f"Skipping character {character.id_number} due to type filter '{character_type_filter}'. Character type is '{character.type}'.", level="DEBUG")
                    continue

                self.append(character)
                success_number += 1
                if logger:
                    logger.gprint(f"Successfully loaded AI character: {character.id_number} ({character.name})", level="INFO")


                # 如果是主要角色
                if character.get_main_character():
                    self.main_characters_id_number.append(character.get_id_number())
                    # Ensure main character influence is initialized, even if no one supports them yet.
                    if character.id_number not in self.main_character_influence:
                        self.main_character_influence[character.get_id_number()] = int(self.initial_influence_score)
                else: # If not a main character, consider it an NPC for this list
                    if character.id_number not in self.npc_characters_id_number:
                         self.npc_characters_id_number.append(character.get_id_number())


            # 登记每个角色和其他所有角色的关系
            # This part seems to be outside the try-except and might run even if character loading failed or was skipped.
            # Moving it inside or ensuring character object is valid.
            # However, the original code has it here, so keeping structure unless it causes direct issues.
            # The `character` variable might not be defined if an error occurred or file was skipped.
            # Let's assume this is intended to run for the last successfully processed `character` from the loop,
            # or it implicitly expects `character` to be valid.
            # For safety, it's better to place this inside the try block or ensure character is valid.
            # Given the current structure, it's safer inside the try block after character is appended.
                relation_items = character.get_relationship().items()
                for target_character_id_number, relation_score in relation_items:
                    if target_character_id_number not in self.relationships[character.get_id_number()]:
                        self.relationships[character.get_id_number()][target_character_id_number] = INITIAL_RELATION_SCORE
                    self.relationships[character.get_id_number()][target_character_id_number] = relation_score
            except Exception as e:
                if logger:
                    logger.gprint(f"Error loading AI character from {character_file}: {e}", level="ERROR")
                else:
                    print(f"Error loading AI character from {character_file}: {e}")
        return success_number

    def get_main_character_influence(self) -> dict:
            relation_items = character.get_relationship().items()
            for target_character_id_number, relation_score in relation_items:
                if target_character_id_number not in self.relationships[character.get_id_number()]:
                    self.relationships[character.get_id_number()][target_character_id_number] = INITIAL_RELATION_SCORE
                self.relationships[character.get_id_number()][target_character_id_number] = relation_score
        return success_number

    def get_main_character_influence(self) -> dict:
        return self.main_character_influence

    def append(self, character: Character) -> None:
        self.character_dict[character.get_id_number()] = character
        self.character_list.append(character)

    def add_character(self, character: Character, logger_passed=None) -> None:
        """
        Adds a new character to the simulation.
        Input:
            character: Character object to be added.
            logger_passed: Optional logger instance.
        """
        if character.get_id_number() in self.character_dict:
            log_msg = f"Character with ID '{character.get_id_number()}' already exists. Cannot add duplicate."
            if logger_passed:
                logger_passed.gprint(log_msg, level="WARNING")
            else:
                print(f"Warning: {log_msg}")
            return

        self.character_list.append(character)
        self.character_dict[character.get_id_number()] = character

        if character.get_main_character():
            if character.get_id_number() not in self.main_characters_id_number:
                self.main_characters_id_number.append(character.get_id_number())
                # Initialize influence for new main character
                self.main_character_influence[character.get_id_number()] = int(self.initial_influence_score)
        else:
            if character.get_id_number() not in self.npc_characters_id_number:
                self.npc_characters_id_number.append(character.get_id_number())
        
        # Re-calculate influence after adding a new character, especially if they support an existing main character
        self.get_influence_for_main_character()

        # Initialize relationships for the new character with all existing characters
        for existing_char in self.character_list:
            if existing_char.get_id_number() != character.get_id_number():
                # New character's relationship with existing character
                if existing_char.get_id_number() not in self.relationships[character.get_id_number()]:
                    self.relationships[character.get_id_number()][existing_char.get_id_number()] = self.initial_relation_score
                # Existing character's relationship with new character
                if character.get_id_number() not in self.relationships[existing_char.get_id_number()]:
                    self.relationships[existing_char.get_id_number()][character.get_id_number()] = self.initial_relation_score
        
        log_msg = f"Character '{character.name}' (ID: {character.id_number}, Type: {character.type}) added to the simulation."
        if logger_passed:
            logger_passed.gprint(log_msg, level="INFO")
        elif hasattr(character, 'logger') and character.logger: # Fallback to character's own logger
            character.logger.gprint(log_msg, level="INFO")
        else:
            print(log_msg)


    def get_character_by_index(self, idx: int) -> Character:
        return self.character_list[idx]

    def get_character_by_id(self, id_number: str) -> Character:
        return self.character_dict.get(id_number, Character())

    def get_all_characters(self, except_for:str=None) -> list:
        if except_for:
            return_list = [i for i in self.character_list if i != except_for]
            if len(return_list) != len(self.character_list):
                print('选择角色时的candidate list已去除',except_for)
            else:
                print('选择角色时的candidate list中未找到',except_for)
            return return_list
        return self.character_list

    def __item__(self, item) -> Character:
        return self.get_character_by_index[item]

    def get_characters_description_except_some(self, except_characters: list) -> str:
        all_character_description = ''
        for character in self.character_list:
            if character.id_number in except_characters: continue
            all_character_description += 'Role ID Number：%s。\n' % character.id_number
        return all_character_description.strip()
