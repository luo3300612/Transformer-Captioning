import torch
import pdb
from collections import defaultdict
import numpy as np
from line_profiler import LineProfiler


def cook_refs_and_test(test, refs, n=4):
    ngram_to_index = {}
    which_gram = []
    info_test = precook(test, ngram_to_index, which_gram, n)
    info_refs = [precook(ref, ngram_to_index, which_gram, n) for ref in refs]
    return info_test, info_refs, ngram_to_index, which_gram


def precook(s, ngram_to_index, which_gram, n=4):
    if isinstance(s, list):
        words = s
    else:
        words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            ngram_index = ngram_to_index.get(ngram, None)
            if ngram_index is None:
                ngram_to_index[ngram] = len(ngram_to_index)
                ngram_index = ngram_to_index[ngram]
                which_gram.append(len(ngram))
            counts[ngram_index] += 1
    return counts, len(words)


def counts2vec(counts, dog_freq, which_gram, global_ref_len, device, n=4):
    counts_tensor = torch.zeros((which_gram.shape[0],), dtype=torch.long, device=device)
    index = torch.tensor(list(counts.keys()), device=device)
    values = torch.tensor(list(counts.values()), device=device)
    counts_tensor.index_put_((index,), values)

    # for k, v in counts.items():
    #     counts_tensor[k] = v
    dog_freq = torch.maximum(dog_freq, torch.ones_like(dog_freq))
    # dog_freq = torch.clip(dog_freq, min=1)
    dog_freq = torch.log(dog_freq)
    vec = counts_tensor * (global_ref_len - dog_freq)
    norm = [torch.linalg.norm(vec[which_gram == i]) for i in range(1, n + 1)]  # TODO further optimize
    return vec, norm


def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref, which_gram, sigma, n):
    delta = float(length_hyp - length_ref)
    val = torch.zeros((4,), device=which_gram.device)
    res_all_grams = torch.minimum(vec_hyp, vec_ref) * vec_ref

    for i in range(1, n + 1):
        tmp_res = torch.sum(res_all_grams[which_gram == i]) / norm_hyp[i - 1] / norm_ref[n - 1]
        tmp_res *= np.e ** (-(delta ** 2) / (2 * sigma ** 2))
        val[i - 1] = tmp_res
    return val


def compute_cider(refs, test, document_frequency, global_ref_len, device, sigma, n=4):
    test_refs_ngrams = []
    for k in refs.keys():
        test_refs_ngrams.append(cook_refs_and_test(test[k][0], refs[k]))
    scores = []
    for info_test, info_refs, ngram_to_index, which_gram in test_refs_ngrams:
        # tmp_doc_freq = torch.zeros((len(ngrams),), dtype=torch.long, device=device)
        cur_doc_freq = torch.zeros((len(ngram_to_index),), device=device)
        for k, v in ngram_to_index.items():
            cur_doc_freq[v] = document_frequency[k]

        which_gram = torch.tensor(which_gram, dtype=torch.long, device=device)
        counts_test, length = info_test
        lp = LineProfiler()
        target_func = lp(counts2vec)

        vec, norm = target_func(counts_test, cur_doc_freq, which_gram, global_ref_len, device)
        lp.print_stats()
        exit(0)

        # print("test vec")
        # print(vec)
        # print("test norm")
        # print(norm)
        score = torch.zeros((4,), device=device)
        for counts_ref, length_ref in info_refs:
            vec_ref, norm_ref = counts2vec(counts_ref, cur_doc_freq, which_gram, global_ref_len, device)
            score += sim(vec, vec_ref, norm, norm_ref, length, length_ref, which_gram, sigma, n)

        score_avg = torch.mean(score)
        score_avg /= len(refs)
        score_avg *= 10
        scores.append(score_avg)

    return torch.mean(torch.stack(scores)), scores
