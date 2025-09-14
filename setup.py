from setuptools import setup, find_packages

setup(
    name="toxicity_deepseek",
    version="0.1",
    py_modules=["Configurations"],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train_model = DAPT.Model_Training:main",
            "generate_dapt = DAPT.Generation:main",
            "train_and_generate = DAPT.Training_and_Generate:main",

            "continuation_generation = Generate_Continuation.Generation:main",
            "DeepSeek_connection = Generate_Continuation.Connection:main",

            "perspective_api = Perspective.Perspective_API:main",
            "generate_prompts = Text_manipulation.Prompt_generation:main",
        ],
    },
)