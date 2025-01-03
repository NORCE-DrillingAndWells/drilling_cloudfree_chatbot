# Cloud-Free Question Answering Chatbot for Drilling Applications

Authors: Liang Zhang, Felix James Pacis, Sergey Alyaev

Year: 2024

**Note, this open repository is work in-progress. We plan to finish the complete instalation instructions during January 2025.**

## Installation

#### 1. Install Ollama

Install an appropriate ollama version using the download instructions: https://ollama.com/download

For linux without admin rights, you can extract the ollama from one of [the pre-compiled releases](https://github.com/ollama/ollama/releases). In our testing, we used the [AMD64 NVIDIA version 0.5.4](https://github.com/ollama/ollama/releases/download/v0.5.4/ollama-linux-amd64.tgz).

#### 2. Download and run an open LLM

Download an LLM for summarization and run it as a service locally.

For llama3.1:8b - small model (--model small), run:

```
ollama run llama3.1:8b
```

For llama3.3 - large model (requires 32+ GB RAM memory) (--model default), run:

```
ollama run llama3.3
```

Keep the service running in the background.

#### 3. Install Git Large File Storage (LFS) to download large weights file

Install Git LFS following [the instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

For Ubuntu linux:

```
apt-get install git-lfs
git lfs install
```

Or, if you do not want to install git-lfs, you can download the weight file `ft_models_v2/6config/model.safetensors` manually and put it in the right place.

#### 4. Download this code

```
git clone https://.../drilling_cloudfree_chatbot.git
```

Ensure the weight file `ft_models_v2/6config/model.safetensors` is the right one.
It can be just a placeholder if you download the code using `git clone` without installing `git-lfs`.
In this case, download the right file manually.

#### 5. Create and activate a virtual enviornment

Create a virtual enviornment:

```
python -m venv chatbotvenv
```

To activate on Windows:

```
chatbotvenv\Scripts\activate
```

and on Linux and MacOS:

```
source chatbotvenv/bin/activate
```

#### 6. Install setuptools

```
pip install setuptools
```

#### 7. Install dependencies from setup.py

```
cd src
pip install .
```

## Usage

#### Run the chatbot

For the chatbot with small model

```
python src/chatbot_demo.py chatbot --model small
```

For testing with default model

```
python src/chatbot_demo.py test --model default
```

## Third party licences:

- The `E5-small-v2` base model is distributed under the [MIT license](https://choosealicense.com/licenses/mit/).
