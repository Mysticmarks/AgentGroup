SAVE_FOLDER = './storage/succession/initial_version'
TEST_FOLDER = './storage/succession/test_version'

# TEST_FOLDER = './storage/succession/saving/chatglm_128k_1'
# TEST_FOLDER = './storage/succession/saving/chatglm_64k_1'
# TEST_FOLDER = './storage/succession/saving/chatglm_1'
# TEST_FOLDER = './storage/succession/saving/llama2_1'
# TEST_FOLDER = './storage/succession/saving/mistral_1'
# TEST_FOLDER = './storage/succession/saving/falcon_1'
# SAVE_FOLDER = TEST_FOLDER
LOG_FOLDER = './logs'
INPUT_FOLDER = './storage/succession/inputs'

model_cache_dir = 'cache_folder'
# Directory where users can store their GGUF model files
GGUF_MODELS_DIR = './gguf_models/'

# Dictionary to register GGUF models with short names.
# Users should uncomment and edit these lines to map short names to their GGUF model file paths.
# Ensure the full path is correct, typically by concatenating GGUF_MODELS_DIR with the filename.
AVAILABLE_GGUF_MODELS = {
    # "llama2-7b-chat": GGUF_MODELS_DIR + "llama-2-7b-chat.Q4_K_M.gguf",
    # "mistral-7b-instruct": GGUF_MODELS_DIR + "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    # Add more models here as needed
}

# Default GGUF model short name to use if a character's 'engine' is not specified or set to "default_gguf".
# Set to one of the keys from AVAILABLE_GGUF_MODELS (e.g., "llama2-7b-chat").
# If set to None or an empty string, characters without a specific engine will not automatically get a GGUF default.
DEFAULT_GGUF_MODEL_NAME = None
# Example:
# DEFAULT_GGUF_MODEL_NAME = "llama2-7b-chat" 

if 'llama2' in TEST_FOLDER:
    model_cache_dir = 'cache_folder/models--meta-llama--Llama-2-7b-chat-hf/snapshots/9eae4b460bfc40df6c741e67d9634f963b31e02e'

debug = False

MIN_SUPPORT_RELATION_SCORE = 20

PRIVATE_CHAT_ROUND = 3
MEETING_CHAT_ROUND = 3
GROUP_CHAT_ROUND = 3
GAME_ROUND = 3

INITIAL_RELATION_SCORE = 10
MAX_RELATION_SCORE_CHANGE = 10
MAX_RELATION_SCORE = 100
MIN_RELATION_SCORE = 0
MAX_BELIEF_SCORE_CHANGE = 10
MIN_BELIEF_SCORE = 0
MAX_BELIEF_SCORE = 100

INITIAL_INFLUENCE_SCORE = 0
MAX_INFLUENCE_SCORE_CHANGE = 30
MAX_INFLUENCE_SCORE = 9999
MIN_INFLUENCE_SCORE = 0

MAX_MEMORY_RETRIEVE_IN_PERCEIVE = 20
MAX_MEMORY_RETRIEVE_IN_REFLECT = 999

ERROR_RETRY_TIMES = 10

ACTIONHISTORY_VOTE = 30
ACTIONHISTORY_NONVOTE = 20
ACTIONHISTORY_RETRIEVE_NUM = 10

ACTIONHISTORY_RETRIEVE_NUM_ANNOUNCEMENT = 20
ACTIONHISTORY_RETRIEVE_NUM_COMPETE = 20
ACTIONHISTORY_RETRIEVE_NUM_COLLABORATE = 20
ACTIONHISTORY_RETRIEVE_NUM_UPDATE = 30
ACTIONHISTORY_RETRIEVE_NUM_WHOLE_INFORMATION = 30
ACTIONHISTORY_RETRIEVE_NUM_PARTIAL_INFORMATION = 30


# ACTIONHISTORY_VOTE = 0
# ACTIONHISTORY_NONVOTE = 0
# ACTIONHISTORY_RETRIEVE_NUM = 0
# ACTIONHISTORY_RETRIEVE_NUM_ANNOUNCEMENT = 0
# ACTIONHISTORY_RETRIEVE_NUM_COMPETE = 0
# ACTIONHISTORY_RETRIEVE_NUM_COLLABORATE = 0
# ACTIONHISTORY_RETRIEVE_NUM_UPDATE = 0
# ACTIONHISTORY_RETRIEVE_NUM_WHOLE_INFORMATION = 30
# ACTIONHISTORY_RETRIEVE_NUM_PARTIAL_INFORMATION = 30
