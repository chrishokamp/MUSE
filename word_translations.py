"""
Method that takes learned cross-lingual embeddings as input and produces word-to-word translations.
"""

import argparse
import torch
import torch.nn.functional as F
import sys
import os
from collections import OrderedDict

from src.models import build_model
from src.utils import get_nn_avg_dist


def get_word_rank_map(fasttext_emb_file):
    """
    Return a map from token-->rank 
    """
     
    word_rank_map = OrderedDict()
    # parse fasttext embedding format
    with open(fasttext_emb_file, 'r') as f:
        # skip the first (header) line
        next(f)

        for i, line in enumerate(f, 1):
            token, _ = line.rstrip().split(' ', 1)
            word_rank_map[token] = i

    return word_rank_map
          

def get_word_translations(emb1, emb2, knn,
                          src_cutoff=None,
                          softmax_temp=1./30.,
                          src_rank_map=None,
                          trg_rank_map=None):
    """
    Given source and target word embeddings, and a list of source words,
    produce a list of lists of k-best translations for each source word.
    """
    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # we always use the contextual dissimilarity measure as this gives the best performance (csls_knn_10)
    # calculate the average distances to k nearest neighbors
    average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
    average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
    average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
    average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)

    top_k_match_ids = []
    step_size = 1000
    if src_cutoff is None:
        src_cutoff = emb1.shape[0]

    for i in range(0, src_cutoff, step_size):
        try:
            print('Processing word ids %d-%d...' % (i, i+step_size))
            word_ids = range(i, i+step_size)

            # use the embeddings of the current word ids
            query = emb1[word_ids]

            # calculate the scores with the contextual dissimilarity measure
            scores = query.mm(emb2.transpose(0, 1))
            scores.mul_(2)
            scores.sub_(average_dist1[word_ids][:, None] + average_dist2[None, :])

            # get the indices of the highest scoring target words
            top_sim_scores, top_match_ids = scores.topk(knn, 1, True)  # returns a (values, indices) tuple (same as torch.topk)
               
            # TODO: if rank heuristic is enabled, filter pairs by the rank heuristic

            # casting to Variable is needed for pytorch 0.3.1 compatibility
            top_sim_scores = F.softmax(softmax_temp * torch.autograd.Variable(top_sim_scores), 1)
             
            top_k_match_ids += [(ids, scores) for ids, scores in zip(top_match_ids, top_sim_scores)]
        except:
            print('Error at index: {}'.format(i))

    return top_k_match_ids


def main(args):
    assert os.path.exists(args.src_emb)
    assert os.path.exists(args.tgt_emb)
    src_emb, tgt_emb, mapping, _ = build_model(args, False)
    src_rank_map = None
    trg_rank_map = None
    if args.word_rank_heuristic:
        rank_window = args.rank_window
        rank_heuristic_threshold = args.rank_heuristic_threshold
        src_rank_map = get_word_rank_map(args.src_emb)
        trg_rank_map = get_word_rank_map(args.tgt_emb)

    # get the mapped word embeddings as vectors of shape [max_vocab_size, embedding_size]
    src_emb = mapping(src_emb.weight).data
    tgt_emb = tgt_emb.weight.data

    id2word1 = {id_: word for word, id_ in args.src_dico.word2id.items()}
    id2word2 = {id_: word for word, id_ in args.tgt_dico.word2id.items()}

    top_k_match_ids = get_word_translations(src_emb, tgt_emb, args.knn,
                                            src_cutoff=args.src_cutoff,
                                            src_rank_map=src_rank_map,
                                            trg_rank_map=trg_rank_map,
                                            softmax_temp=args.softmax_temp)

    output_file = '%s-%s.txt' % (args.src_lang, args.tgt_lang)
    print('Writing to %s...' % output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        for src_id, (tgt_ids, tgt_scores) in enumerate(top_k_match_ids):
            for tgt_id, score in zip(tgt_ids, tgt_scores):
                if args.cuda:
                    tgt_id, score = tgt_id.cpu(), score.cpu()
                if args.output_scores:

                    checks_passed = True
                    try:                    
                        if args.word_rank_heuristic:
                            src_rank = src_rank_map[id2word1[src_id]] 
                            trg_rank = trg_rank_map[id2word2[tgt_id]] 
                            if src_rank < rank_heuristic_threshold and abs(src_rank - trg_rank) >= rank_window:
                                checks_passed = False
                        
                        if checks_passed:
                            # torch 0.4.0
                            #f.write('%s %s %.4f\n' % (id2word1[src_id],
                            #                          id2word2[int(tgt_id.numpy())],
                            #                          float(score.numpy())))
                            # TODO: this is a temporary hack -- the correct way would be to apply filter heuristic before softmax
                            f.write('%s %s %.9f\n' % (id2word1[src_id],
                                                      id2word2[tgt_id],
                                                      float(score)))
                    except:
                        pass 
                else:
                    # torch 0.4.0
                    #f.write('%s %s\n' % (id2word1[src_id],
                    #                     id2word2[int(tgt_id.numpy())]))
                    f.write('%s %s\n' % (id2word1[src_id],
                                         id2word2[tgt_id]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src-lang', required=True, help='Source language')
    parser.add_argument('--tgt-lang', required=True, help='Target language')
    parser.add_argument('-s', '--src-emb', required=True, help='The path to the source language embeddings')
    parser.add_argument('-t', '--tgt-emb', required=True, help='The path to the target language embeddings')
    parser.add_argument('--max-vocab', type=int, default=500000, help='Maximum vocabulary size')
    parser.add_argument('--emb-dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--src_cutoff', type=int, default=None, help='If specified, dictionary will only be output up to this rank of src')
    parser.add_argument('--cuda', action='store_true', help='Run on GPU')
    parser.add_argument("--normalize_embeddings", type=str, default='', help="Normalize embeddings before training")
    parser.add_argument("--output_scores", dest='output_scores', action='store_true',
                        help="Whether normalized scores should be included in output")
    parser.set_defaults(output_scores=False)
    parser.add_argument("--word_rank_heuristic", dest='word_rank_heuristic', action='store_true',
                        help="whether to filter the pairs using the word rank heuristic")
    parser.set_defaults(word_rank_heuristic=False)
    parser.add_argument("--rank_window", type=int, required=False, default=7500,
                        help="if word_rank_heuristic is enabled, the size of the rank window that matches can be selected from")
    parser.add_argument("--rank_heuristic_threshold", type=int, default=100000,
                        help="if word_rank_heuristic is enabled, the heuristic will only be applied to words below this rank")
    parser.add_argument('--softmax_temp', type=float, default=1./30.,
                        help='Softmax temperature for score normalization')
    parser.add_argument('--knn', type=int, default=10,
                        help='K-NNs that should be retrieved for each source word'
                             '(Conneau et al. use 10 for evaluation)')
    args = parser.parse_args()
    if not args.cuda:
        print('Not running on GPU...', file=sys.stderr)

    main(args)
