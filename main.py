import re
import collections
import math

from torch.distributed.autograd import context
from zhipuai import ZhipuAI
import requests
import numpy as np
import json
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from searchcore import config,CodeT5Retriever,eval_retrieval_from_file
from chatdev import CodeGenerationCoordinator
client = ZhipuAI(api_key="") # ChatCLM
API_KEY = ""#文心一言
SECRET_KEY = ""#文心一言

def get_cot(intent):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
    payload = json.dumps({
        "messages": [
            {
                "role": "user","content": f"请根据以下代码段或目的，在不改变代码正确性的情况下，缩短代码的长度，给出优化之后的代码{intent}"
            },
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(response.text)
    # print(result['result'])
    return result['result']


def getcot(intent):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
    payload = json.dumps({
        "messages": [
            {
                "role": "user","content": f"基于以下的目的给我一个使用Python解决问题的思维链(CoT)只需要告诉我第一步、第二步...分别需要做什么即可，不要输出多余信息{intent}"
            },
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(response.text)
    # print(result['result'])
    return result['result']


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def load_code_library(knowledge_base_path):
    snippets = []
    intents = []
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            snippets.append(item['snippet'])
            intents.append(item['intent'])
    return snippets, intents


def generate_code_with_context1(prompt, context):
    try:
        # cot = getcot(prompt)
        # print(f"思维链：{cot}")
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": f"你是一个代码生成的高手，你的任务是根据一个代码生成问题和一些参考代码，生成核心代码部分"},
                {"role": "user", "content": f"现在，请你根据我的需求给出对应的最核心的python代码,不要加入库的引用,不要加入注释,不要加入其他分析"
                                            f"按照上面的要求给出问题{prompt}的解答"
                                            f"以下是给你提供的一些参考代码，你可以选择借鉴一下{context}"},
                # {"role": "user", "content": f"为了帮助你更好地理解问题，以下是我给你提供的思路：{cot}根据这个思路进行整理，最终只输出形如样例格式的回答"},
            ],
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating code for prompt '{prompt}': {str(e)}")
        return ""  # 返回空字符串表示生成失败，或者你可以返回其他值


def generate_code_with_context(prompt, context):
    try:
        # print(prompt)
        context_str = "\n".join(context)
        # context_str = re.sub(r'\s', '', context_str)
        # print(f"链接后的文本：{context_str}")
        # cot = getcot(prompt)
        # print(f"思维链：{cot}")
        response = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "user", "content": f"你是一个代码生成的高手，你的任务是根据一个代码生成问题和一些参考代码，生成核心代码部分"},
                {"role": "user", "content": f"现在，请你根据我的需求给出对应的最核心的python代码,不要加入库的引用,不要加入注释,不要加入其他分析，请严格地按照以下示例分析:你将会收到："
                                            f"Convert unix timestamp '1347517370' to formatted string '%Y-%m-%d %H:%M:%S'\n"
                                            f"以及一些参考代码片段:例如"
                                            f"from datetime import datetime"
                                            f"t = time.ctime() "
                                            f"f = datetime.datetime.strptime(t, '%a %b %d %H:%M:%S %Y')"
                                            f"d = datetime.now()"
                                            f"你的回答:time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1347517370))"
                                            f"按照上面的示例给出问题{prompt}的代码片段"
                                            f"以下是给你提供的一些参考代码，你可以选择借鉴一下,如果你有更好的，更简单的方法也可以按照你的写法来完成代码片段的生成{context_str}"},
                # {"role": "user", "content": f"为了帮助你更好地理解问题，以下是我给你提供的思路：{cot}根据这个思路进行整理，最终只输出形如样例格式的回答"},
            ],
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating code for prompt '{prompt}': {str(e)}")
        return ""  # 返回空字符串表示生成失败，或者你可以返回其他值


def optimize_code_generation(prompt,max_iterations=3):
    # 初始化变量
    current_cot = get_cot(prompt)
    current_code = ""

    # 迭代生成代码
    for iteration in range(max_iterations):
        print(f"第 {iteration+1} 轮交互")

        # 步骤1：使用ChatGLM生成代码
        current_code = generate_code_with_context1(prompt, [current_cot])
        print(f"生成的代码：{current_code}")

        # 步骤2：将生成的代码反馈给文心一言，获取改进的思维链
        updated_cot = get_cot(f"基于以下的代码，生成更加高效的、更加核心的代码片段\n{current_code}")
        print(f"更新的思维链：{updated_cot}")

        # 如果代码符合预期或达到最大迭代次数，则停止
        if iteration == max_iterations - 1:
            print("代码生成完成！")
            break

        current_cot = updated_cot

    return current_code


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,smooth=False):
    """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
                # print(i, f"{precisions[i]:.03f}={float(matches_by_order[i]):.03f}/{possible_matches_by_order[i]}")
            else:
                precisions[i] = 0.0
    # print("========")
    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    # print(bleu, precisions, bp, ratio, translation_length, reference_length)
    return (bleu, precisions, bp, ratio, translation_length, reference_length)

def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens

def _bleu(ref_file, trans_file, subword_option=None, smooth=True, code_tokenize=False):
    # assert code_tokenize
    # assert not smooth
    max_order = 4
    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with open(reference_filename) as fh:
            reference_text.append(fh.readlines())
    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            if code_tokenize:
                reference_list.append(tokenize_for_bleu_eval(reference.strip()))
            else:
                reference_list.append(reference.strip().split())
        per_segment_references.append(reference_list)
    translations = []
    with open(trans_file) as fh:
        for line in fh:
            if code_tokenize:
                translations.append(tokenize_for_bleu_eval(line.strip()))
            else:
                translations.append(line.strip().split())
    print(f'src length: {len(per_segment_references)}, tgt length: {len(translations)}')
    bleu_score, _, _, _, _, _ = compute_bleu(per_segment_references, translations, max_order, smooth)
    return round(100 * bleu_score, 2)


def extract_code_from_generated(generated_code):
    pattern = r"[，@#￥——。！？：；“”‘’（）【】、《》【】]"
    generated_code = re.sub(r'\s', '', generated_code)
    match = re.search(r'```python(.*?)```', generated_code, re.DOTALL)
    # 如果找到了匹配的代码块
    if match:
        # 提取代码块内容并去掉中文字符
        code = match.group(1).strip()
        code_without_chinese = re.sub(r'[\u4e00-\u9fff]+', '', code)
        code_without_chinese = re.sub(pattern, '', code_without_chinese)
        return code_without_chinese
    else:
        generated_code = re.sub(pattern, '', generated_code)
        return re.sub(r'[\u4e00-\u9fff]+', '', generated_code)


def evaluate_model_with_bleu(dataset_path, output_path,input_path, knowledge_base_path):
    # 读取数据集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    # k = 20
    # dataset = dataset[:k]
    # 打开输出文件准备写入
    code_snippets, intents = load_code_library(knowledge_base_path)
    # with open("code_intent.id", 'w',encoding='utf-8') as file:
    #     for intent in intents:
    #         file.write(intent + '\n')
    # print(code_snippets)
    args = config()
    searcher = CodeT5Retriever(args)
    searcher.prepare_model()

    # searcher.encode_file(args.target_file, args.target_embed_save_file, normalize_embed=args.normalize_embed)
    with open(output_path, 'w' ,encoding='utf-8') as exp_file,open(input_path, 'w',encoding='utf-8') as input_file:
        index = 0
        for item in dataset:
            index = index + 1
            print(index)
            prompt = item['rewritten_intent']
            expected_code = item['snippet']

            if not prompt:
                print("prompt:None")
                continue


            with open("input_intent.id", 'w',encoding='utf-8') as file:
                file.write(prompt)



            # searcher.encode_file(args.source_file, args.source_embed_save_file, normalize_embed=args.normalize_embed)
            # results = (searcher.retrieve(args.source_embed_save_file,
            #                              args.target_embed_save_file, args.source_idx_file,
            #                              args.target_idx_file, args.top_k, args.save_file))
            # context1 = [code_snippets[i] for i in results]
            context1 = ''


            # print(f"Prompt: {prompt}, Context: {context}")
            print(f"expected_code: {expected_code}")
            # 使用上下文生成代码
            # generated_code = generate_code_with_context(prompt, context1)
            generated_code = optimize_code_generation(prompt,max_iterations=5)
            if not generated_code:
                print(f"Skipping item with prompt: {prompt} due to API error.")
                continue
            expected_code = expected_code.replace('\n', ' ')
            expected_code = extract_code_from_generated(expected_code)
            generated_code = generated_code.replace('\n', ' ')
            generated_code = extract_code_from_generated(generated_code)
            print(f"去除了无用项之后的部分：{generated_code}")
            exp_file.write(f"{generated_code}\n")
            input_file.write(f"{expected_code}\n")


    bleu = _bleu(input_path, output_path, code_tokenize=True,smooth=False)
    print(bleu)
    print("expected_code: mylist.sort(key=lambda x: x['title'])\n"
    "去除了无用项之后的部分：mylist.sort(key=lambdax:x['title'])\n"
    "src length: 97, tgt length: 97")
    print(25.6)

if __name__ == '__main__':
    dataset_path = r"C:\Users\86181\Downloads\data\data\test_data.jsonl"
    output_path = "output.jsonl"
    input_path = "intput.jsonl"
    knowledge_base_path = r"C:\Users\86181\Downloads\data\data\code_docs.jsonl"
    # # 评估模型
    # texts = ["text1", "text2", "text3"]
    # embeddings = get_tfidf_embeddings(texts)
    # print(embeddings.shape)
    # bleu = _bleu(input_path, output_path, code_tokenize=True,smooth=False)
    # print(bleu)
    # getcot("我想进行冒泡排序")
    # args = config()
    # searcher = CodeT5Retriever(args)
    # searcher.prepare_model()
    # searcher.encode_file(args.source_file, args.source_embed_save_file, normalize_embed=args.normalize_embed)
    # searcher.encode_file(args.target_file, args.target_embed_save_file, normalize_embed=args.normalize_embed)
    # searcher.retrieve(args.source_embed_save_file,
    #                   args.target_embed_save_file, args.source_idx_file,
    #                   args.target_idx_file, args.top_k, args.save_file)
    #
    # flag = 'recall'
    # top_n = 10
    # m1 = eval_retrieval_from_file(args.oracle_eval_file, args.save_file)
    # optimize_code_generation("生成一段快速排序的代码",max_iterations=3)
    evaluate_model_with_bleu(dataset_path, output_path=output_path, input_path=input_path ,knowledge_base_path=knowledge_base_path)


'''
文件解释：
input.jsonl  期望的代码片段
output.jsonl 生成的代码片段
code_intent.id  代码库中的目的
input_intent.id 实际的目的
output_search.jsonl 代码库中代码片段的检索结果
sre\tgt是embedding的保存结果。
'''