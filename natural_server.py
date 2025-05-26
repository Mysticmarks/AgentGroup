import asyncio
import os
import json # Keep json import
from typing import List, Dict, Optional, Any # Keep typing imports

from fastapi import FastAPI # Keep FastAPI for startup event
import uvicorn

from natural_conversation_core import ConversationManager, CoreMessage 
from character.all_character_class import AllCharacter
from character.character_class import Character 
from config import SAVE_FOLDER, LOG_FOLDER, DEFAULT_GGUF_MODEL_NAME # Keep config imports
from logger_class import Logger

# --- Initialize Logger ---
# Ensure LOG_FOLDER has a default if it might be None or empty from config
effective_log_folder = LOG_FOLDER if LOG_FOLDER else "./logs/"
os.makedirs(effective_log_folder, exist_ok=True)
main_server_logger = Logger(effective_log_folder, logger_name="SimplifiedNaturalWebServerLogger")
main_server_logger.gprint("Simplified Natural Web Server Logger initialized.")

# --- Initialize ConversationManager and Load Participants ---
# Corrected constructor call to use logger_instance
global_conversation_manager = ConversationManager(logger_instance=main_server_logger)

try:
    if not SAVE_FOLDER or not os.path.exists(SAVE_FOLDER) or not os.path.isdir(SAVE_FOLDER):
        main_server_logger.gprint(f"Warning: SAVE_FOLDER ('{SAVE_FOLDER}') is not configured or does not exist. No scenario characters will be loaded.", error=True, level="WARNING")
        all_participants_loader = AllCharacter(save_folder=None, logger=main_server_logger)
    else:
        all_participants_loader = AllCharacter(save_folder=SAVE_FOLDER, logger=main_server_logger)
        
    initial_participants_ids = []
    if all_participants_loader.get_all_characters(): # Check if any characters were loaded
        for char_obj in all_participants_loader.get_all_characters():
            global_conversation_manager.add_participant(char_obj.id_number, char_obj)
            initial_participants_ids.append(char_obj.id_number)
    main_server_logger.gprint(f"Loaded initial participants into ConversationManager: {initial_participants_ids}")

    carl_id = "user_ai_carl_web_simplified" 
    if carl_id not in global_conversation_manager.participants:
        # Ensure default_gguf or another valid engine is configured
        carl_engine = DEFAULT_GGUF_MODEL_NAME if DEFAULT_GGUF_MODEL_NAME else "default_gguf_placeholder"
        carl = global_conversation_manager.create_and_add_agent(
            id_name=carl_id, name="Carl (Web Simplified)",
            objective="To chat via simplified server.", scratch="AI for simplified web chat.",
            background="An AI for simplified web.", engine=carl_engine 
        )
        if carl: main_server_logger.gprint(f"Dynamically added '{carl_id}'.")
        else: main_server_logger.gprint(f"Failed to add '{carl_id}'.", error=True, level="ERROR")
except Exception as e:
    main_server_logger.gprint(f"Critical error during participant loading: {e}", error=True, level="CRITICAL")

# --- FastAPI App Initialization (minimal for startup event) ---
app = FastAPI()

# --- Background Task for ConversationManager processing ---
async def run_conversation_processing_loop():
    main_server_logger.gprint("Conversation processing background task starting...")
    # Ensure global_conversation_manager is accessible if it was potentially set to None in except block
    if global_conversation_manager: 
        while True:
            try:
                global_conversation_manager.process_message_queue()
            except Exception as e:
                main_server_logger.gprint(f"Error in global_conversation_manager.process_message_queue: {e}", error=True, level="ERROR")
            await asyncio.sleep(0.2) # Interval for processing
    else:
        main_server_logger.gprint("Conversation processing background task cannot start: global_conversation_manager is None.", error=True, level="CRITICAL")


@app.on_event("startup")
async def startup_event():
    main_server_logger.gprint("FastAPI startup: Initializing conversation processing background task.")
    asyncio.create_task(run_conversation_processing_loop())

# --- Main execution block ---
if __name__ == '__main__':
    # LOG_FOLDER should be handled by the logger init itself now.
    # os.makedirs(LOG_FOLDER, exist_ok=True) # Already handled by effective_log_folder logic
    main_server_logger.gprint(f"Starting Uvicorn server (simplified) on 0.0.0.0:8000 for natural_server.py")
    uvicorn.run(app, host='0.0.0.0', port=8000)
