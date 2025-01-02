import os, sys
# , ollama
import requests, json
import retriever
import time
import argparse

TOP_RETRIEVALS_TO_USE = 5

current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
sys.path.append(current_folder + "/..")

Test_Query = "Where was the well 2/6-3 targeting?"


def generate_prompt(query: str, index_passage_pair: dict, topX: int) -> str:
    context = retriever.get_context(query, topX, index_passage_pair)
    prompt = (f"Question: {query}. "
              f"Answer the question **directly to the best of your ability** based solely on the provided context. "
              f"Note that not all the context is related to the question. "
              f"Do not include any explanation or additional information. "
              f""
              f"Context: {context}")
    return prompt, context


def ask_LLM_ollama(prompt: str, cleanAnswer: bool = False, fixedSeed = False, smallModel=False) -> str:
    # Define the URL and the data payload
    url = "http://localhost:11434/api/generate"
    model_name = "llama3.3"
    # Check if smallModel is boolean type and true
    if isinstance(smallModel, bool) and smallModel:
        model_name = "llama3.1:8b"
    elif isinstance(smallModel, str):
        model_name = smallModel
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
        # "format": "json",
    }
    if fixedSeed:
        data['options'] = {
            "seed": 0,
            "temperature": 0
        }

    # Send the POST request
    response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

    # Check the response status and print the result
    if response.status_code == 200:
        if cleanAnswer:
            print("Response:\n", response.json()["response"], "\n")
        else:
            print("Response:\n", response.json(), "\n")
    else:
        print("Error:", response.status_code, response.text, "\n")


def ask_LLM_openai(prompt: str, cleanAnswer: bool = False) -> str:
    API_URL = "https://api.openai.com/v1/chat/completions"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": f"{prompt}"}],
        "stream": False,
        # "temperature": 1
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))

    # Check the response status and print the result
    if response.status_code == 200:
        if cleanAnswer:
            print("Response:\n", response.json()["choices"][0]["message"]["content"], "\n")
        else:
            print("Response:\n", response.json(), "\n")
    else:
        print("Error:", response.status_code, response.text, "\n")
    return response


def test(smallModel=False):
    query_list = [
        "When was well 16/10-5 abandoned?",
        "When was well 6204/10-2R abandoned?",
        "When was well 1/2-1 drilled in Norway?",
        "Where was the well 2/6-3 targeting?",
        "What was the objective of the well 32/4-2?",
        "What was the CO2 storage potential discovered by 32/4-2?",
        "What was the CO2 storage potential discovered by 32/4-3S?",
        "Which reservoir is the well 33/2-1 drilled into?",
        "Which geology types were observed in well 6201/11-3R?",
        "Which wells targeted Late and Middle Jurassic sandstones?",
        "What was the primary objective of well 34/6-3 A?",
        "What were the primary and secondary objectives of well 30/7-8 R?",
        "What is the source of information for the well named 7/11-14A?",
        "What were the results of the drill stem test DST 3 in the Ekofisk Formation?",
        "What was the main purpose of drilling well 16/2-21?",
    ]

    with open(current_folder + "/index_passage_pair.json", "r") as file:
        index_passage_pair = json.load(file)
    # prompt = generate_prompt(query_str, index_passage_pair, 3)
    # print("Prompt:\n", prompt + "\n")
    # ask_LLM_openai(prompt)
    for i in range(len(query_list)):
        query_str = query_list[i]

        prompt, context = generate_prompt(query_str, index_passage_pair, TOP_RETRIEVALS_TO_USE)
        print("Prompt:\n", prompt + "\n")


        start_time = time.time()  # Record start time
        ask_LLM_ollama(prompt, cleanAnswer=True, fixedSeed=True, smallModel=smallModel)
        end_time = time.time()  # Record end time

        print(f"Got response in {end_time - start_time}")



def chatbot(smallModel=False):
    print("Welcome to the CLI Chatbot! Type 'exit' to end the conversation.")
    with open(current_folder + "/index_passage_pair.json", "r") as file:
        index_passage_pair = json.load(file)
    while True:
        print("You:")
        user_input = input("")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        prompt, context = generate_prompt(user_input, index_passage_pair, TOP_RETRIEVALS_TO_USE)
        print("Prompt:\n", prompt + "\n")
        # print("Context:")
        # for c in context:
        #     print(c)
        ask_LLM_ollama(prompt, cleanAnswer=True, fixedSeed=True, smallModel=smallModel) 
        # ask_LLM_openai(prompt, True)


if __name__ == "__main__":
    # parse the arguments and run test if test is given as the first argument
    # the second argument can point at a small model "mosel=[model]" to be used or "model=small" to use the default small model
    parser = argparse.ArgumentParser(description="Run the chatbot or test mode.")
    parser.add_argument("mode", choices=["chatbot", "test"], default='chatbot', help="Mode to run the script in.", nargs="?")
    parser.add_argument("--model", default="default", help="Specify the model to use.")
    args = parser.parse_args()

    small_model = False
    if args.model == "small":
        small_model = True
    elif args.model != "default":
        small_model = args.model
    if args.mode == "test":
        test(smallModel=small_model)
    else:
        chatbot(smallModel=small_model)

