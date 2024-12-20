import os, sys

current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
sys.path.append(current_folder + "/..")

import numpy as np
import json

import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch import Tensor


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_e5_embeddings(input_text: str, pretrained_model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    model = AutoModel.from_pretrained(pretrained_model_name_or_path)
    batch_dict = tokenizer(
        input_text, max_length=512, padding=True, truncation=False, return_tensors="pt"
    )  # no truncation
    try:
        outputs = model(**batch_dict)
        print(input_text[:80])
    except:
        batch_dict = tokenizer(
            input_text, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )  # no truncation
        outputs = model(**batch_dict)
        print(input_text[:80], "truncated")

    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # reframe the embeddings shape from torch.Size([1, 1024]) to ndarray shape (1024,)
    embeddings = embeddings[0].detach().numpy()

    return embeddings


def cosine_similarity(vec1, vec2):
    # Calculate the dot product between the two vectors
    dot_product = np.dot(vec1, vec2)

    # Calculate the magnitude of each vector
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)

    # Calculate the cosine similarity
    cosine_sim = dot_product / (magnitude_vec1 * magnitude_vec2)

    return cosine_sim


def get_passage_by_id(id: str, index_passage_pair: dict = None) -> str:
    if index_passage_pair == None:
        file_path = current_folder + "/index_passage_pair.json"
        with open(file_path, "r") as file:
            index_passage_pair = json.load(file)
    passage = index_passage_pair[id]
    return passage


def generate_index_passage_pair(save_to_file: bool = True) -> dict:
    file_path = current_folder + "/../data_to_encode_v2/raw_passage_id_to_passage_for_embedding.jsonl"
    index_passage_pair = {}
    with open(file_path, "r") as file:
        for line in file:
            # Parse each line as a JSON object
            data = json.loads(line)
            index_passage_pair.update({data["text_id"]: data["text"]})
    if save_to_file:
        with open(current_folder + "/index_passage_pair.json", "w") as file:
            json.dump(index_passage_pair, file, indent=4)
    return index_passage_pair


def get_index_for_queryEmbedding(queryEmbedding: list | np.ndarray, topX: int) -> dict:
    with open(current_folder + "/index_embedding_pair.json", "r") as file:
        index_embedding_pair = json.load(file)
    index_list = list(index_embedding_pair.keys())
    embedding_list = list(index_embedding_pair.values())
    sim_score_list = []
    for i in range(len(index_list)):
        embedding_vec = embedding_list[i]
        sim_score = cosine_similarity(queryEmbedding, embedding_vec)
        sim_score_list.append(sim_score)

    # Combine the lists into a list of tuples
    combined_list = list(zip(sim_score_list, index_list))
    # Sort the combined list by sim_scores_list in descending order
    sorted_combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)
    # Unzip the sorted list back into two lists
    sorted_sim_score_list, sorted_index_list = zip(*sorted_combined_list)
    # Convert the tuples back to lists
    sorted_sim_score_list = list(sorted_sim_score_list)
    sorted_index_list = list(sorted_index_list)

    result_dict = {}
    for i in range(topX):
        index = sorted_index_list[i]
        sim_score = float(round(sorted_sim_score_list[i], 5))
        # passage = get_passage_by_id(sorted_index_list[i])
        result_dict.update({i: {"index": index, "sim_score": sim_score}})
    print(result_dict, "\n")
    return result_dict


def get_context(query_str: str, topX: int, index_passage_pair: dict) -> list:
    pretrained_model_path = current_folder + "/../ft_models_v2/6config"

    emb_vec_query = get_e5_embeddings(query_str, pretrained_model_path)
    # print("embedding vector of query: ", emb_vec_query)
    index_dict = get_index_for_queryEmbedding(emb_vec_query, topX)
    context = []
    for i in index_dict.keys():
        index = index_dict[i]["index"]
        passage = get_passage_by_id(index, index_passage_pair)
        context.append(passage)
    return context


def get_index_for_queryEmbedding_chroma(queryEmbedding: list, topX: int) -> dict:
    pass


if __name__ == "__main__":
    string_toEmbed = "hello world!"
    # embedding = ce.get_embedding(string_toEmbed, "intfloat/e5-small-v2")
    # print(embedding)

    # pretrained_model_path = current_folder + "/../ft_models_v2/6config"
    # emb_vec_query = get_e5_embeddings(string_toEmbed, pretrained_model_path)
    # print("embedding vector of query: ", emb_vec_query)

    vec1 = np.array(
        [
            0.00988967,
            -0.131895959,
            0.366581589,
            -0.386717469,
            -0.185908973,
            -0.308382481,
            0.436351895,
            -0.876174212,
            0.164197415,
            -0.516570389,
            -0.0129162595,
            -0.43401444,
            0.0231531151,
            0.14578408,
            0.0549943596,
            0.0208770037,
            -0.101431936,
            0.248535573,
            -0.492506117,
            0.070788756,
            0.109564118,
            0.149367303,
            0.545031488,
            -0.480692059,
            0.423635185,
            0.271412134,
            -0.41422385,
            -0.0526967645,
            -0.403042346,
            -1.52925146,
            -0.187208757,
            0.308895618,
            0.297455341,
            -0.745426238,
            0.263229191,
            -0.750637531,
            -0.172030032,
            0.00341193378,
            0.512997508,
            0.575055957,
            -0.266469926,
            0.533780456,
            -0.116552547,
            0.051086992,
            0.0716240183,
            -0.0147104114,
            -0.468472,
            -0.0861276314,
            0.467860103,
            -0.150943905,
            -0.151547313,
            -0.318646401,
            0.463415265,
            -0.351137936,
            0.784860671,
            0.570648253,
            0.204889745,
            0.085978657,
            0.551952362,
            0.491529942,
            -0.0826776326,
            0.447656035,
            -1.00740981,
            0.735173106,
            0.610446393,
            0.1914002,
            -0.0612528957,
            0.205643401,
            -0.630103409,
            -0.197767794,
            -0.0080992775,
            0.760910153,
            0.0206161626,
            0.177621931,
            -0.849368513,
            -0.349888146,
            -0.245909691,
            0.124673545,
            -0.241572857,
            -0.0410446972,
            0.0953508168,
            -0.683304548,
            0.487572104,
            -0.0275898129,
            -0.348949492,
            0.452067673,
            0.196702,
            -0.0302161574,
            0.12154834,
            0.256759197,
            -0.322756857,
            0.613754928,
            0.381687939,
            -0.476939976,
            -0.630282283,
            0.0446637273,
            0.145628497,
            0.0104378425,
            0.397695065,
            1.25454926,
            0.589231193,
            -0.100099817,
            0.41592139,
            -0.266155899,
            -0.547310948,
            -0.0255183801,
            0.0966151357,
            -0.242779687,
            0.5049,
            -0.186081588,
            0.0568086505,
            -0.137022674,
            0.0428231806,
            -0.277877063,
            -0.167551041,
            -0.0204929933,
            0.0743893087,
            0.0756415129,
            -0.432089269,
            0.187122345,
            0.402409732,
            -0.282918572,
            -0.158967033,
            0.283291936,
            0.217061758,
            -0.936529577,
            0.36623311,
            0.416580886,
            0.318847775,
            -0.247727841,
            0.442425489,
            -0.146459565,
            -0.104850098,
            -0.0640866905,
            -0.206457049,
            0.530899525,
            -0.6695171,
            -0.0487722829,
            0.188401863,
            0.026723275,
            -0.423776269,
            -0.567182839,
            0.428204596,
            -0.602486253,
            -0.291332752,
            0.550292,
            -0.627298057,
            0.162650034,
            -0.45223093,
            -0.175934866,
            -0.424431324,
            0.213959187,
            0.281943351,
            0.298521489,
            0.0731446,
            -0.595659494,
            -0.193708807,
            -0.339100212,
            -0.498332739,
            -0.00320727937,
            -0.562711477,
            -0.417313933,
            0.552220345,
            0.797635913,
            -0.220085353,
            -0.657239079,
            0.596758366,
            -0.0810331628,
            0.195920378,
            -0.12650016,
            0.386703134,
            0.712922812,
            -0.102639571,
            0.0264636017,
            1.02734315,
            -0.250176936,
            -0.154838294,
            0.30262962,
            0.101888523,
            -0.750782132,
            0.441503942,
            0.360221654,
            0.273373753,
            0.637488,
            0.467994243,
            -0.0548205599,
            -0.564430356,
            -0.198600948,
            0.174960896,
            -0.291549414,
            -0.559257,
            0.16201967,
            -0.0211004429,
            0.124658845,
            0.00414640456,
            0.78338021,
            -0.108598441,
            -0.0391319059,
            -0.210863203,
            -0.0995869711,
            0.240774602,
            0.269776583,
            0.00099428,
            -0.24172464,
            -0.491250306,
            0.10298039,
            0.0783387,
            -0.173576832,
            0.000642968342,
            -0.106016621,
            0.272635639,
            -0.263005286,
            -0.0120256655,
            -0.716722608,
            -0.904123783,
            -0.526243925,
            0.130404621,
            0.291727185,
            -0.0619022734,
            -0.316671759,
            0.904237509,
            -0.425979018,
            -0.831900716,
            -1.28586495,
            0.489627928,
            -0.293656081,
            -0.201937109,
            0.0504487,
            -0.588965714,
            0.0192751363,
            -0.0329162851,
            -0.60271579,
            0.0678197816,
            0.503710568,
            -0.580193162,
            -0.092126742,
            -0.370107949,
            -0.201664329,
            0.542992651,
            -0.130918473,
            -0.141117647,
            0.470246851,
            -0.603150606,
            0.284422755,
            0.00998129696,
            -0.335175335,
            -0.810202122,
            -0.0183753949,
            0.199929312,
            0.447545201,
            0.0580461472,
            0.459856361,
            -0.661827,
            -0.0426239967,
            0.250713915,
            0.470791727,
            -0.395937026,
            0.909381866,
            -0.147487894,
            0.228908658,
            0.355373532,
            -0.381685317,
            -0.277940571,
            0.167030737,
            0.205497369,
            0.46577996,
            -0.474876583,
            0.313342214,
            0.215361178,
            -0.76283592,
            0.542402148,
            -0.0250656959,
            0.0620077401,
            0.220738217,
            -0.667931199,
            -0.184066206,
            -0.259292364,
            -0.333989143,
            0.141135931,
            -0.160736531,
            0.200984478,
            0.0893636942,
            0.547376573,
            0.146831602,
            -0.0641354471,
            0.859928608,
            -0.0933008045,
            0.261173844,
            -0.108783118,
            0.200282872,
            -0.211740896,
            0.104812726,
            -0.0264372732,
            0.153531089,
            0.205322474,
            -0.293145329,
            0.259178579,
            0.378216147,
            0.119985342,
            0.304809391,
            -0.442687571,
            0.111103505,
            -0.247889057,
            0.150875017,
            -0.218659908,
            -0.319767654,
            0.230811566,
            0.157021269,
            0.424293,
            0.195155472,
            -0.211038351,
            0.150809631,
            0.021569401,
            0.608810604,
            -0.420192927,
            -0.242584899,
            -0.0433289111,
            0.135812894,
            -0.0512471572,
            -1.5492357,
            -0.400084615,
            0.278554231,
            0.236331552,
            -0.533225656,
            0.1765991,
            0.312010318,
            -0.354360521,
            -0.533012509,
            0.112170726,
            -0.0720817223,
            0.312778205,
            0.69321084,
            -0.214865535,
            -0.387615442,
            0.0303302668,
            0.579372704,
            0.235832885,
            -0.364612192,
            -0.0320790708,
            0.0362266265,
            0.390314579,
            1.36359727,
            0.108683072,
            -0.173780456,
            0.0782977194,
            0.249197662,
            -0.285898805,
            0.0486222953,
            0.130069926,
            -0.292449027,
            -0.0774809048,
            0.295228153,
            -0.0104478542,
            0.0714106411,
            0.588515699,
            -0.193553,
            0.431352973,
            0.0421305932,
            -0.673469,
            0.154003546,
            0.310081363,
            -0.241575152,
            -0.723352432,
            0.822562,
            0.270061046,
            -0.492662728,
            -0.350077778,
            0.407136589,
            -0.703274727,
            0.0637666881,
            0.111722127,
            0.558182359,
            0.533168,
            0.040329434,
            -0.553154826,
            -0.279736876,
            -0.472408354,
            0.219309017,
            -0.347425818,
            0.223023534,
            -0.332813114,
            -0.130710736,
            -0.0192481261,
            0.767027378,
        ]
    )
    get_index_for_queryEmbedding(vec1, 10)
