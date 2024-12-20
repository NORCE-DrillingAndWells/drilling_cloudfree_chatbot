import os, sys
import requests, json
import retriever

current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
sys.path.append(current_folder + "/..")


def generate_prompt(query: str, index_passage_pair: dict, topX: int) -> str:
    context = retriever.get_context(query, topX, index_passage_pair)
    prompt = f"Question: {query}. Answer the question based solely on the provided context. Note that not all the context is related to the question. Do not include any explanation or additional information. Context: {context}"
    return prompt


def ask_LLM_ollama(prompt: str, cleanAnswer: bool = False) -> str:
    # Define the URL and the data payload
    # url = "http://10.64.105.44:11434/api/generate"
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False,
        # "format": "json",
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


def test():
    query_list = [
        "What was the primary objective of well 34/6-3 A?",
        "What were the primary and secondary objectives of well 30/7-8 R?",
        "What is the source of information for the well named 7/11-14A?",
        "What were the results of the drill stem test DST 3 in the Ekofisk Formation?",
        "What was the main purpose of drilling well 16/2-21?",
    ]
    query_str = query_list[1]
    with open(current_folder + "/index_passage_pair.json", "r") as file:
        index_passage_pair = json.load(file)
    # prompt = generate_prompt(query_str, index_passage_pair, 3)
    # print("Prompt:\n", prompt + "\n")
    # ask_LLM_openai(prompt)
    prompt = generate_prompt(query_str, index_passage_pair, 3)
    print("Prompt:\n", prompt + "\n")
    ask_LLM_ollama(prompt)


def chatbot():
    print("Welcome to the CLI Chatbot! Type 'exit' to end the conversation.")
    with open(current_folder + "/index_passage_pair.json", "r") as file:
        index_passage_pair = json.load(file)
    while True:
        print("You:")
        user_input = input("")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        prompt = generate_prompt(user_input, index_passage_pair, 3)
        print("Prompt:\n", prompt + "\n")
        ask_LLM_ollama(prompt, True)
        # ask_LLM_openai(prompt, True)


if __name__ == "__main__":
    # test()
    chatbot()
