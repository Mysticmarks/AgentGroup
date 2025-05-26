import json
import contextvars
import uvicorn
from typing import List, Optional # Added Optional
from pydantic import BaseModel
from uuid import uuid4
from fastapi import FastAPI, BackgroundTasks, status, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os # Ensure os is imported for SAVE_FOLDER, LOG_FOLDER
import asyncio # For potential async tasks with ConversationManager

from config import * # SAVE_FOLDER, LOG_FOLDER will be from here
from config import AVAILABLE_GGUF_MODELS # Import for the new endpoint
from logger_class import Logger # Changed from Logger_v2
from main import SucArena # This seems specific to the old game, might not be needed for chat
from resources.info_bank import * # Might not be needed for chat
from help_functions import * # Might not be needed for chat

from natural_conversation_core import ConversationManager # Import ConversationManager
from character.all_character_class import AllCharacter # To load participants
from character.character_class import Character # For type hinting if needed, though ConversationManager uses it


# 默认全局变量
log_dir = LOG_FOLDER # Used by main_server_logger
# save_folder = SAVE_FOLDER # Used by AllCharacter loader

# Context局部变量 - These seem related to the old game structure, may not be directly used by ConversationManager
# context_sid = contextvars.ContextVar('sid') # Retaining for now, but new chat features may not use it
# context_test_folder = contextvars.ContextVar('test_folder') # Retaining for now

app = FastAPI()

# Pydantic model for the new agent creation request from the web UI
class NewAgentWebRequest(BaseModel):
    id_name: str
    name: str
    objective: str
    scratch: str
    background: str
    engine: str
    character_type: Optional[str] = "ai"
    is_main_character: Optional[bool] = False
    portrait_path: Optional[str] = None
    small_portrait_path: Optional[str] = None
    # Beliefs, relations, judgements are complex and might be better handled by default or a more advanced UI
    # For now, we'll let ConversationManager.create_and_add_agent use its defaults for these if not provided

# --- Global Logger and Conversation Manager Setup ---
try:
    main_server_logger = Logger(LOG_FOLDER, log_level="INFO") # Logger for server-specific messages
    main_server_logger.gprint("Main server logger initialized.", level="INFO")

    # Load participants using AllCharacter
    # SAVE_FOLDER should be defined in config.py and point to the scenario to load
    # e.g., "./storage/succession/initial_version"
    if not os.path.exists(SAVE_FOLDER) or not os.path.isdir(SAVE_FOLDER):
        main_server_logger.gprint(f"Error: SAVE_FOLDER '{SAVE_FOLDER}' not found or not a directory. Please check config.py. Cannot load characters.", level="CRITICAL")
        # Depending on server requirements, might exit or run with no pre-loaded characters
        all_participants_loader = None # Indicate failure or skip loading
    else:
        all_participants_loader = AllCharacter(save_folder=SAVE_FOLDER, logger=main_server_logger)

    # Initialize ConversationManager, passing the socket_manager instance for broadcasting
    global_conversation_manager = ConversationManager(
        logger_instance=main_server_logger, 
        socket_broadcaster=socket_manager # Pass the server's socket_manager
    )

    active_human_id_for_server: Optional[str] = None
    if all_participants_loader:
        if all_participants_loader.get_all_characters():
            for char_obj in all_participants_loader.get_all_characters():
                global_conversation_manager.add_participant(char_obj.id_number, char_obj)
                main_server_logger.gprint(f"[Server Setup] Added participant: {char_obj.id_number} ({char_obj.name})", level="INFO")
            
            # Identify a human for WebSocket messages (e.g., the first human found)
            for char_id, char_obj in global_conversation_manager.participants.items():
                if char_obj.type == "human":
                    active_human_id_for_server = char_id
                    main_server_logger.gprint(f"[Server Setup] Identified human for WebSocket default actions: {active_human_id_for_server} ({char_obj.name})", level="INFO")
                    break
            
            if not active_human_id_for_server and global_conversation_manager.participants:
                # Fallback if no human, pick first available (though this might not be desired for web client)
                # active_human_id_for_server = list(global_conversation_manager.participants.keys())[0]
                # main_server_logger.gprint(f"[Server Setup] No human participant found. Defaulting actions to first participant: {active_human_id_for_server}", level="WARNING")
                main_server_logger.gprint(f"[Server Setup] No human participant found. WebSocket messages will be attributed to a generic ID if no human is designated.", level="WARNING")

        else:
            main_server_logger.gprint(f"[Server Setup] No characters loaded by AllCharacter from {SAVE_FOLDER}. ConversationManager is empty.", level="WARNING")
    else: # all_participants_loader was None due to SAVE_FOLDER issue
        main_server_logger.gprint(f"[Server Setup] AllCharacter loader not initialized. ConversationManager is empty.", level="WARNING")


except Exception as e:
    print(f"CRITICAL ERROR during server setup: {e}")
    # Optionally, re-raise or exit if setup is vital.
    global_conversation_manager = None 
    if 'main_server_logger' not in locals(): # If logger init itself failed
        main_server_logger = Logger(LOG_FOLDER, log_level="ERROR") # Fallback logger
    main_server_logger.gprint(f"CRITICAL ERROR during server setup: {e}. ConversationManager might not be functional.", level="CRITICAL", error=True) # type: ignore

if global_conversation_manager:
    # Set the WebSocket broadcast callback for the ConversationManager
    global_conversation_manager.set_websocket_broadcast_callback(socket_manager.broadcast)

    # Dynamically create user_ai_carl if he doesn't exist (from previous main block)
    # This ensures 'user_ai_carl' is available for conversation.
    if "user_ai_carl" not in global_conversation_manager.participants:
        main_server_logger.gprint("Attempting to dynamically create 'user_ai_carl'...", level="INFO")
        carl = global_conversation_manager.create_and_add_agent(
            id_name="user_ai_carl",
            name="Carl (Web AI)",
            objective="To observe and learn from web users via WebSocket.",
            scratch="A newly created AI curious about web conversations, connected via server.",
            background="Carl is an AI for the web, interacting through server.py.",
            engine="default_gguf" # Make sure DEFAULT_GGUF_MODEL_NAME is set in config
        )
        if carl:
            main_server_logger.gprint(f"Agent '{carl.name}' (ID: {carl.id_number}) dynamically added to manager.", level="INFO")
        else:
            main_server_logger.gprint("Failed to dynamically create and add 'user_ai_carl'.", level="ERROR")

# --- End Global Logger and Conversation Manager Setup ---

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections: # Ensure websocket is in list before removing
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

socket_manager = ConnectionManager()

# 配置跨域
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=False,
	allow_methods=["*"],
	allow_headers=["*"],
)


class GameConfig(BaseModel):
    game_round: int = GAME_ROUND
    battle_chat_round: int = BATTLE_CHAT_ROUND
    collabration_chat_round: int = COLLABORATION_CHAT_ROUND
    sid: str = ""


@app.post("/api/v1/create")
async def create_session(setting: GameConfig, background_tasks: BackgroundTasks):    
    # 获取sid
    if setting.sid:  # 如果sid不为空，读档
        # 如果sid不存在，返回报错信息
        all_sids = [sid for sid in os.listdir(TEST_FOLDER)]
        if setting.sid not in all_sids:
            raise HTTPException(status_code=404, detail="sid not exist")
        # 设置当前sid
        context_sid.set(setting.sid)
    else:  # 否则开始新游戏
        context_sid.set(str(uuid4().hex))

    # 获取logger
    cur_logger = await Logger_v2.create(log_dir, context_sid.get())
    conetxt_logger.set(cur_logger)

    # 获取test_folder
    cur_test_folder = os.path.join(TEST_FOLDER, context_sid.get())
    if not os.path.exists(cur_test_folder):
        # 如果test_folder不存在，创建test_folder，初始内容为SAVE_FOLDER中的内容
        os.makedirs(cur_test_folder)
        copy_dir(SAVE_FOLDER, cur_test_folder)
    context_test_folder.set(cur_test_folder)
    
    background_tasks.add_task(start_game, setting)
    return {"sid": context_sid.get()}


def start_game(setting: GameConfig):
    sucarena = SucArena(all_round_number=setting.game_round,
                        battle_chat_round=setting.battle_chat_round,
                        collabration_chat_round=setting.collabration_chat_round,
                        save_folder=context_test_folder.get(),
                        test_folder=context_test_folder.get(),
                        logger=conetxt_logger.get())
    try:
        logger = conetxt_logger.get()
        test_folder = context_test_folder.get()

        for i in range(setting.game_round):
            logger.gprint('==' * 10)
            logger.gprint('Turn %d' % (i + 1),
                          important_log='important_log',
                          source_character='',
                          target_character='',
                          log_type='Turn Change',
                          thought='',
                          log_content='Turn %d' % (i + 1))

            logger.gprint('==' * 10)
            logger.gprint('Confrontation Stage',
                          important_log='important_log',
                          source_character='',
                          target_character='',
                          log_type='Stage Change',
                          thought='',
                          log_content='Confrontation Stage')
            sucarena.compete_stage(i)

            logger.gprint('==' * 10)
            logger.gprint('Cooperation Stage',
                          important_log='important_log',
                          source_character='',
                          target_character='',
                          log_type='Stage Change',
                          thought='',
                          log_content='Cooperation Stage')
            sucarena.collaborate_stage(i)

            logger.gprint('==' * 10)
            logger.gprint('Announcement Stage',
                          important_log='important_log',
                          source_character='',
                          target_character='',
                          log_type='Stage Change',
                          thought='',
                          log_content='Announcement Stage')
            sucarena.announcement_stage(i)

            logger.gprint('==' * 10)
            logger.gprint('Update Stage',
                          important_log='important_log',
                          source_character='',
                          target_character='',
                          log_type='Stage Change',
                          thought='',
                          log_content='Update Stage')
            sucarena.update_stage(i)

            logger.gprint('Start Saving')
            sucarena.save(test_folder)

            logger.gprint('==' * 10)
            logger.gprint('',
                          important_log='important_log',
                          source_character='',
                          target_character='',
                          log_type='Turn End',
                          thought='',
                          log_content='')

        # 结算阶段
        game_name = 'Succession'

        logger.gprint('==' * 10)
        logger.gprint('Settlement Turn',
                      important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Turn Change',
                      thought='',
                      log_content='Settlement Turn')
        local_information_winner = sucarena.settlement_stage(whole_information=False, game_name=game_name)
        logger.gprint('',
                      important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Turn End',
                      thought='',
                      log_content='')

        logger.gprint('==' * 10)
        logger.gprint('Settlement Turn (Cheating)',
                      important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Turn Change',
                      thought='',
                      log_content='Settlement Turn (Cheating)')
        whole_information_winner = sucarena.settlement_stage(whole_information=True, game_name=game_name)
        logger.gprint('',
                      important_log='important_log',
                      source_character='',
                      target_character='',
                      log_type='Turn End',
                      thought='',
                      log_content='')

        logger.gprint('Start Saving')
        sucarena.save(test_folder)
        logger.gprint('Game ends successfully')
        logger.close()
    except Exception as e:
        logger.gprint(e)
        logger.gprint('Game ends with error')
        logger.close()


@app.get("/api/v1/get")
async def get_logs(sid: str, last: int):
    """
    Retrieve logs of session with sid.

    INPUT:
        sid: session id
        last: the last retrieved log id
    OUTPUT:
        important logs from (last + 1)
    """
    from_id = last + 1
    log_file = os.path.join(log_dir, f"{sid}.json")
    lines = await read_logs(log_file, from_id)
    results = []
    for line in lines:
        line = eval(text_translation(line, id_trans_table))  # 替换角色名后转成字典
        line_common = {'sid': line['sid'], 'id': line['id'], 'time': line['time'], 'args': line['args']}
        line_kwargs = eval(line['kwargs'])
        results.append({**line_common, **line_kwargs})
    return results


async def read_logs(log_file, from_id):
    with open(log_file, 'r', encoding='utf-8') as log_fw:
        lines = log_fw.readlines()[from_id:]
    return lines


class UserInput(BaseModel):
    sid: str = ""
    input_str: str = ""


@app.post("/api/v1/input")
async def get_input(user_input: UserInput):
    if not os.path.exists(INPUT_FOLDER):
        os.mkdir(INPUT_FOLDER)
    # 如果sid不存在，返回报错信息
    all_sids = [sid.split('.txt')[0] for sid in os.listdir(INPUT_FOLDER)]
    if user_input.sid not in all_sids:
        raise HTTPException(status_code=404, detail="sid not exist")
    # 写入文件
    save_input_path = os.path.join(INPUT_FOLDER, user_input.sid + ".txt")
    with open(save_input_path, "w", encoding="utf-8") as f:
        f.write(user_input.input_str)


@app.get("/api/v1/quicksimulate")
async def quick_simulate():
    """
    获取筛选后的sid用于快速simulate

    OUTPUT:
        {
            "Simulation1": "2e295fa3cddd47e8bccbc377608cf179",
            "Simulation2": "be141b8d5e2c4befb34e2cf358dd705e",
            "Simulation3": "c81e3621c50640aba0aad3feae9dae7c"
        }
    """
    return quick_simulate_sids


@app.get("/api/v1/getsettings")
async def get_settings():
    """
    角色和社会资源展示

    OUTPUT:
        {
            "characters": [{character1 details}, {character2 details}, ...],
            "resources": [{resource1 details}, {resource2 details}, ...]
        }
    """
    characters, resources = [], []
    save_characters_folder = os.path.join(save_folder, "characters")
    save_resources_folder = os.path.join(save_folder, "resources")
    for character_file in os.listdir(save_characters_folder):
        f_path = os.path.join(save_characters_folder, character_file)
        with open(f_path, 'r', encoding='utf-8') as f:
            character_detail = json.load(f)
        characters.append(character_detail)
    for resource_file in os.listdir(save_resources_folder):
        f_path = os.path.join(save_resources_folder, resource_file)
        with open(f_path, 'r', encoding='utf-8') as f:
            resource_detail = json.load(f)
        resources.append(resource_detail)
    return {"characters": characters, "resources": resources}


class NewCharacterRequest(BaseModel):
    name: str = ""
    main_character: str = ""
    support_character: str = ""
    objective: str = ""
    scratch: str = ""
    background: str = ""
    belief: List[int] = []  # 5个信念分
    relation: List[int] = []  # 与9个默认角色的关系分
    portrait: str = ""
    small_portrait: str = ""


@app.post("/api/v1/addcharacter", status_code=status.HTTP_201_CREATED)
async def add_character(request: NewCharacterRequest):
    """
    插入新角色

    INPUT: NewCharacterRequest
        name: str
        main_character: str
        support_character: str
        objective: str
        scratch: str
        background: str
        belief: List[int] = []  # 5个信念分
        relation: List[int] = []  # 与9个默认角色的关系分
        portrait: str
        small_portrait: str
    """
    # get new index
    save_characters_folder = os.path.join(save_folder, "characters")
    max_idx = 0
    for character_file in os.listdir(save_characters_folder):
        curr_idx = int(character_file.split("C")[1].split(".json")[0])
        if curr_idx > max_idx:
            max_idx = curr_idx
    new_idx = "C" + str(max_idx + 1).zfill(4)

    # some check
    # check support character in all characters
    all_characters = [character.split('.json')[0] for character in os.listdir(save_characters_folder)]
    if request.support_character not in all_characters:
        raise HTTPException(status_code=404, detail="Support character not exist")
    # check belief and relation score num
    if len(request.belief) != len(all_characters) or len(request.relation) != len(all_characters):
        raise HTTPException(status_code=500, detail="Element number of belief or relation not equal to all character number")

    # create new character
    character = {
        "name": request.name,
        "id_name": new_idx,
        "main_character": request.main_character,
        "support_character": request.support_character,
        "objective": request.objective,
        "scratch": request.scratch,
        "background": request.background,
        "belief": {f"Stand with C000{i}": request.belief[i] for i in range(len(request.belief))},
        "judgement": {},
        "relation": {f"C000{i}": request.relation[i] for i in range(len(request.relation))},
        "portrait": request.portrait,
        "small_portrait": request.small_portrait
    }
    # save new character
    save_path = os.path.join(save_characters_folder, new_idx + ".json")
    with open(save_path, 'w', encoding='utf-8') as f:    
        json_data = json.dumps(character, indent=4, ensure_ascii=False)    
        f.write(json_data)
    return "New Character Created"


class NewResourceRequest(BaseModel):
    name: str = ""
    description: str = ""
    influence: str = ""
    owner: str = ""
    topic: List[str] = []
    portrait: str = ""
    small_portrait: str = ""


@app.post("/api/v1/addresource", status_code=status.HTTP_201_CREATED)
async def add_resource(request: NewResourceRequest):
    """
    插入新社会资源 

    INPUT: NewResourceRequest
        name: str
        description: str
        influence: str
        owner: str
        topic: str
        portrait: str
        small_portrait: str
    """
    # get new index
    save_resources_folder = os.path.join(save_folder, "resources")
    max_idx = 0
    for resource_file in os.listdir(save_resources_folder):
        curr_idx = int(resource_file.split("R")[1].split(".json")[0])
        if curr_idx > max_idx:
            max_idx = curr_idx
    new_idx = "R" + str(max_idx + 1).zfill(4)

    # some check
    # check if influence is a number
    try:
        influence_score = int(request.influence)
    except Exception:
        raise HTTPException(status_code=500, detail="Influnce score should be a integer")
    # check owner in all characters
    save_characters_folder = os.path.join(save_folder, "characters")
    all_characters = [character.split('.json')[0] for character in os.listdir(save_characters_folder)]
    if request.owner not in all_characters:
        raise HTTPException(status_code=404, detail="Owner not exist")

    # create new resource
    resource = {
        "name": request.name,
        "id_number": new_idx,
        "description": request.description,
        "influence": request.influence,
        "owner": request.owner,
        "topic": "[TOPIC_SEP]".join(request.topic),
        "portrait": request.portrait,
        "small_portrait": request.small_portrait
    }
    # save new resource
    save_path = os.path.join(save_resources_folder, new_idx + ".json")
    with open(save_path, 'w', encoding='utf-8') as f:    
        json_data = json.dumps(resource, indent=4, ensure_ascii=False)    
        f.write(json_data)
    return "New Resource Created"


@app.get("/test")
def test():
    return "hello"


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await socket_manager.connect(websocket)
    # Optionally, notify other clients that a new user has joined.
    # The ConversationManager's broadcast via callback will handle general message display.
    # Specific "user joined" messages could be added here or in ConnectionManager.connect if desired.
    client_id_log = f"{websocket.client.host}:{websocket.client.port}"
    main_server_logger.gprint(f"WebSocket client {client_id_log} connected.", level="INFO")

    # Determine speaker_id for this connection.
    # For simplicity, use active_human_id_for_server if available, else a generic ID.
    # A more robust system might involve authentication or selection from the client.
    speaker_id_for_this_ws_connection = active_human_id_for_server if active_human_id_for_server else f"web_user_{client_id_log}"
    if active_human_id_for_server:
        main_server_logger.gprint(f"WebSocket client {client_id_log} will submit messages as participant '{speaker_id_for_this_ws_connection}'.", level="INFO")
    else:
        main_server_logger.gprint(f"No specific human participant mapped for WebSocket client {client_id_log}. Messages will be submitted as '{speaker_id_for_this_ws_connection}'. This ID may not be a registered Character.", level="WARNING")


    try:
        while True:
            data = await websocket.receive_text()
            main_server_logger.gprint(f"Raw message from WebSocket client {client_id_log}: {data}", level="DEBUG")

            if global_conversation_manager:
                try:
                    # Assuming the client sends JSON: {"userId": "actual_user_id", "content": "message"}
                    # However, the instructions for websocket_test.html imply it sends raw text.
                    # For now, let's assume the client sends raw text and use `speaker_id_for_this_ws_connection`.
                    # If JSON is required, the client JS needs to change.
                    # Let's use the simpler raw text input for now, as per websocket_test.html.
                    
                    # The instruction "Parse the JSON, Extract userId and content" contradicts client.
                    # Adhering to current client that sends raw text:
                    received_content = data 
                    user_id_for_submission = speaker_id_for_this_ws_connection

                    # Validate if the user_id_for_submission is a known participant before submitting.
                    # This is crucial if speaker_id_for_this_ws_connection is just a generic ID.
                    if user_id_for_submission not in global_conversation_manager.participants:
                        # This case is problematic if "web_user_..." is used and not a real participant.
                        # submit_message in ConversationManager has a warning for this.
                        # For a stricter system, we might reject if not a known participant.
                        main_server_logger.gprint(f"Warning: Submitting message from '{user_id_for_submission}' who might not be a fully registered participant if it's a generic ID.", level="WARNING")

                    global_conversation_manager.submit_message(
                        speaker_id=user_id_for_submission, 
                        content=received_content
                    )
                    # process_message_queue is now run by the background task, so no direct call here.
                    
                except Exception as e_inner: # Catch errors during message submission
                    main_server_logger.gprint(f"Error processing message from client {client_id_log} ('{data}'): {e_inner}", level="ERROR", error=True) # type: ignore
                    await websocket.send_text("System: Error processing your message.")
            else:
                main_server_logger.gprint("Error: global_conversation_manager not initialized. Cannot process WebSocket message.", level="ERROR")
                await websocket.send_text("System: Server error, cannot process message.") # Inform client

    except WebSocketDisconnect:
        socket_manager.disconnect(websocket)
        main_server_logger.gprint(f"WebSocket client {client_id_log} disconnected.", level="INFO")
    except Exception as e: 
        main_server_logger.gprint(f"Error with WebSocket client {client_id_log}: {e}", level="ERROR", error=True) # type: ignore
        socket_manager.disconnect(websocket)

# Background task to run ConversationManager's processing loop
async def run_conversation_processing_loop():
    if not global_conversation_manager:
        main_server_logger.gprint("Conversation processing loop cannot start: ConversationManager not initialized.", level="CRITICAL")
        return
    
    main_server_logger.gprint("Starting background conversation processing loop...", level="INFO")
    while True:
        try:
            global_conversation_manager.process_message_queue()
        except Exception as e:
            main_server_logger.gprint(f"Error in conversation processing loop: {e}", level="ERROR", error=True) # type: ignore
        await asyncio.sleep(0.1) # Adjust sleep time as needed

@app.on_event("startup")
async def startup_event():
    # Start the conversation processing loop as a background task
    asyncio.create_task(run_conversation_processing_loop())


@app.post("/api/v1/create_agent", status_code=status.HTTP_201_CREATED)
async def create_agent_endpoint(agent_data: NewAgentWebRequest):
    if not global_conversation_manager:
        raise HTTPException(status_code=503, detail="ConversationManager not available.")
    
    main_server_logger.gprint(f"Received request to create agent: {agent_data.id_name}", level="INFO")
    
    # Convert Pydantic model to dict for create_and_add_agent, handling None values
    agent_params = agent_data.model_dump(exclude_none=True)

    # create_and_add_agent expects specific parameters, so we map them.
    # It also has defaults for character_type and is_main_character.
    created_agent = global_conversation_manager.create_and_add_agent(
        id_name=agent_params.get("id_name"),
        name=agent_params.get("name"),
        objective=agent_params.get("objective"),
        scratch=agent_params.get("scratch"),
        background=agent_params.get("background"),
        engine=agent_params.get("engine"),
        character_type=agent_params.get("character_type", "ai"), # Use Pydantic default if not in payload
        is_main_character=agent_params.get("is_main_character", False), # Use Pydantic default
        portrait_path=agent_params.get("portrait_path"),
        small_portrait_path=agent_params.get("small_portrait_path")
        # Beliefs, relations, judgements are not included in NewAgentWebRequest for simplicity,
        # create_and_add_agent will use defaults.
    )
    
    if created_agent:
        return {"message": f"Agent '{created_agent.name}' created successfully.", "agent_id": created_agent.id_number}
    else:
        # create_and_add_agent logs errors internally
        raise HTTPException(status_code=400, detail=f"Failed to create agent '{agent_data.id_name}'. Check server logs for details.")

@app.get("/api/v1/get_gguf_models")
async def get_gguf_models_endpoint():
    return AVAILABLE_GGUF_MODELS


if __name__ == '__main__':
    uvicorn.run(app='server:app', host='0.0.0.0', port=8080, reload=True)