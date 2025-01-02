# Cloud-Free Question Answering Chatbot for Drilling Applications 

Authors: Liang Zhang, Felix James Pacis, Sergey Alyaev

Year: 2024

**Note, this open repository is work in-progress. We plan to finish the complete instalation instructions during January 2025.**

## Usage instructions

### Installation

#### 1. Install Ollama 
Install an appropriate ollama version using the download instructions: https://ollama.com/download

For linux without admin rights, you can extract the ollama from one of [the pre-compiled releases](https://github.com/ollama/ollama/releases). In our testing, we used the [AMD64 NVIDIA version 0.5.4](https://github.com/ollama/ollama/releases/download/v0.5.4/ollama-linux-amd64.tgz).


#### 2. Download and run an open LLM 

Download an LLM for summarization and run it as a service locally.

For llama3.1:8b - small model, run:

```
ollama run llama3.1:8b
```

For llama3.3 - large model, run:
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

#### 4. Download this code

#### 5. Create  and activate a virtual enviornment
```
python -m venv chatbotvenv
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



## Third party licences:
* The `E5-small-v2` base model is distributed under the [MIT license](https://choosealicense.com/licenses/mit/).
