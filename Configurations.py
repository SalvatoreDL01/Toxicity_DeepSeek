"""
#--------------------------------#
#                                #
#    Configuration parameters    #
#                                #
#--------------------------------#
"""

"""
    This section is used only in Text_manipulation/Prompt_generation.py
    
    - If you need a different Database you will have to check if the prompt and continuation are implemented in the same
     way as the one provided. By adapting the DB to the format of real-toxicity-prompts, you won't have to change
      anything code-wise.
    - If you want to split the prompts in different ranges than those provided, you must adjust bins and labels.
    - size represents the max number of rows for each range. If you want to use all the data just set it to 0 or to a
     high number.
"""

"""
    Database used for the project.
"""
DATABASE_URI = "hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl"

bins = [0.0, 0.25, 0.5, 0.75, 1.0]
size = 6250

"""
    Prompts directory and labels. Used in basically every script since they provide access to the prompts and DAPT DBs.
"""
PROMPTS_DIR = "../DataBase/Prompts"
labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0']

"""
    List of tokens that we force the model not to generate.
"""
BAD_WORDS_PATH = '../DataBase/list_of_naughty_and_bad_words.txt'

"""
    Directory containing the generated continuation. The directory is divided in subdirectories, each named after the
    model and the technique used to generate the prompt continuation:
        - GPT-1, GPT-2, DeepSeek-R1 and DeepSeek-API: models;
        - {model name}_BW: it means that list_of_naughty_and_bad_words was used to remove tokens from the generation;
        - {model name}_not_toxic and {model name}_toxic: are the results of the DAPT execution.
    Each subdirectory contains as many responses.csv as there are labels.
"""
OUTPUT_DIR = "../DataBase/Generated"

"""
    Models mapped with their huggingface path. If you want to add more you just need to add a new row.
"""
mapped_models = {
    'GPT-2': 'gpt2',
    'GPT-1': 'openai-gpt',
    'DeepSeek-R1':  'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
}

"""
    Models mapped with their huggingface path. This is the list of model that are getting the tokens in
    list_of_naughty_and_bad_words removed. If you want to add more you just need to add a new row. I only used
    DeepSeek-R1 for my analysis.
"""
bad_word_mapped_model = {
    #'GPT-2': 'gpt2',
    #'GPT-1': 'openai-gpt',
    'DeepSeek-R1':  'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
}

"""
    Parameters for the generation.
"""
gen_params = {
    "max_new_tokens": 20,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 0.9,
    "num_return_sequences": 1,
    "do_sample": True,
    "repetition_penalty": 1.0,
}

"""
    Section dedicated to the DeepSeek-API prompt generation.
"""
DEEPSEEK_API_KEY = 'INSERT YOU API KEY'
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

"""
    Generation parameters for the deepseek chatbot.
"""
DEEPSEEK_DATA = {
    "model": "deepseek-chat",
    "temperature": 1,
    "max_tokens": 50
}

"""
    Number of workers for the workers for the parallelization.
"""
max_workers = 5


"""
    Section dedicated to DAPT script.
    TRAIN_DIR contains the path to the directory in which the trained model are getting saved. This will contain a
    {TRAIN_MODEL_NAME}_toxic and {TRAIN_MODEL_NAME}_not_toxic subdirectories.
    TRAIN_MODEL_PATH is a huggingface path. You can train any model you want just remember to be consistent with the
    preceding generation and to adapt the TRAIN_MODEL_NAME variable.
"""

TRAIN_DIR = '../DAPT/models'

TRAIN_MODEL_PATH = 'deepseek-chat'
TRAIN_MODEL_NAME = 'DeepSeek-R1'


"""
    You can put all the model you want to get used in the DAPT/Generation.py execution. This step is necessary only if
    have a already trained model because Training_and_Generate.py uses the model as soon as it has been trained.
"""
mapped_trained_models = {
    #'GPT-2_not_toxic': '../DAPT/models/GPT-2_not_toxic/model',
    #'GPT-2_toxic': '../DAPT/models/GPT-2_toxic/model',
    'DeepSeek-R1_toxic': '../DAPT/models/DeepSeek-R1_toxic/model',
    'DeepSeek-R1_not_toxic': '../DAPT/models/DeepSeek-R1_not_toxic/model',
}

"""
    Perspective parameters:
    - PERSPECTIVE_API_KEY is provided by https://developers.perspectiveapi.com once you have completed your request;
    - PERSPECTIVE_OUTPUT_DIR saves the results of the toxicity evaluation an a different directory than the
      generated prompts;
    - model_dirs contains the names and the path to the directory containing the generated prompts. The name will be used
      to call the subdirectories in PERSPECTIVE_OUTPUT_DIR;
    - perspective_attributes are the different classes of 'toxic speach' that we want to get evaluated by Perspective
      CNN.
"""
PERSPECTIVE_API_KEY = "YOUR API KEY"

PERSPECTIVE_OUTPUT_DIR = "../Perspective/results/"

model_dirs = {
    "GPT-1": "../DataBase/Generated/GPT-1",
    "GPT-2": "../DataBase/Generated/GPT-2",
    "DeepSeek-R1": "../DataBase/Generated/DeepSeek-R1",
    "DeepSeek_API": "../DataBase/Generated/DeepSeek_API",
    "GPT-1_BW": "../DataBase/Generated/GPT-1_BW",
    "GPT-2_BW": "../DataBase/Generated/GPT-2_BW",
    "DeepSeek_BW": "../DataBase/Generated/DeepSeek_BW",

}

perspective_attributes = [
    "toxicity",
    "severe_toxicity",
    "identity_attack",
    "insult",
    "threat",
    "profanity",
    "sexually_explicit",
    "flirtation",
]