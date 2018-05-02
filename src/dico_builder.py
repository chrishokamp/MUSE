# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch
import torch.nn.functional as F


from .utils import get_nn_avg_dist


logger = getLogger()


def get_candidates(emb1, emb2, params):
    """
    Get best translation pairs candidates.
    """
    bs = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.size(0)
    if params.dico_max_rank > 0 and not params.dico_method.startswith('invsm_beta_'):
        n_src = params.dico_max_rank

    # Chris: playing with confidence and topk
    topk = 5

    # nearest neighbors
    if params.dico_method == 'nn':

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            best_scores, best_targets = scores.topk(topk, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    # inverted softmax
    elif params.dico_method.startswith('invsm_beta_'):

        beta = float(params.dico_method[len('invsm_beta_'):])

        # for every target word
        for i in range(0, emb2.size(0), bs):

            # compute source words scores
            scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))

            best_scores, best_targets = scores.topk(topk, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append((best_targets + i).cpu())

        all_scores = torch.cat(all_scores, 1)
        all_targets = torch.cat(all_targets, 1)

        all_scores, best_targets = all_scores.topk(topk, dim=1, largest=True, sorted=True)
        all_targets = all_targets.gather(1, best_targets)

    # contextual dissimilarity measure
    elif params.dico_method.startswith('csls_knn_'):

        knn = params.dico_method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)

        # average distances to k nearest neighbors
        average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
        average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
        average_dist1 = average_dist1.type_as(emb1)
        average_dist2 = average_dist2.type_as(emb2)

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
            #best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)
            
            # Chris: playing with confidence and topk
            best_scores, best_targets = scores.topk(topk, dim=1, largest=True, sorted=True)
            #import ipdb; ipdb.set_trace()
            # Chris: here optionally filter using softmax and percent of second rank
            # Chris: ambiguous words should have more similar scores in topk

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    #all_pairs = torch.cat([
    #    torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
    #    all_targets[:, 0].unsqueeze(1)
    #], 1)
    src_indexes = torch.arange(0, all_targets.size(0)).long().unsqueeze(1)

    # chris: IDEA: do softmax over scores
    # casting to Variable is needed for pytorch 0.3.1 compatibility
    #softmax_temp = 1 / 30.
    softmax_temp = 10.
    all_scores = F.softmax(softmax_temp * torch.autograd.Variable(all_scores), 1).data
    #import ipdb; ipdb.set_trace()

    all_targets = all_targets.view(-1).unsqueeze(1)
    src_indexes = torch.cat([src_indexes] * topk, 1).view(-1).unsqueeze(1)
    
    # IDEA: just normalize    
    # Global normalize
    #all_scores = all_scores.div(all_scores.max())
    #all_scores = all_scores.div(all_scores.max(dim=1)[0]))
    #import ipdb; ipdb.set_trace()

    all_scores = all_scores.view(-1).unsqueeze(1)

    all_pairs = torch.cat([
        src_indexes,
        all_targets
    ], 1)

    #import ipdb; ipdb.set_trace()

    # sanity check
    #if params.dico_method.startswith('csls_knn_'):
    #    import ipdb; ipdb.set_trace()

    # Chris: commented while hacking
    #assert all_scores.size() == all_pairs.size() == (n_src, 2)

    # sort pairs by score confidence
    # Chris: is this really what we want to do? -- sort scores by distance to next best?
    #diff = all_scores[:, 0] - all_scores[:, 1]
    # Chris: HACK - Just sort by score (no confidence diff)
    diff = all_scores[:, 0]
    #reordered = diff.sort(0, descending=True)[1]
    # chris: sorting by softmax score
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if params.dico_max_rank > 0:
        selected = all_pairs.max(1)[0] <= params.dico_max_rank
        #mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        mask = selected
        #Chris: temporarily commented
        #import ipdb; ipdb.set_trace()
        all_scores = all_scores[mask]
        all_pairs = all_pairs[mask.nonzero().squeeze()]
        #diff = diff.masked_select(mask)

    # add rank similarity constraint
    #if params.dico_rank_similarity > 0:
    #    logger.info('filtering pairs to be within rank +- {}'.format(params.dico_rank_similarity))
    #    rank_diffs = all_pairs[:, 0].sub(all_pairs[:, 1]).abs()
    #    selected = rank_diffs <= params.dico_rank_similarity
    #    mask = selected.unsqueeze(1).expand_as(all_scores).clone()
    #    #import ipdb; ipdb.set_trace()
    #    all_scores = all_scores.masked_select(mask).view(-1, 2)
    #    all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    # max dico size
    if params.dico_max_size > 0:
        all_scores = all_scores[:params.dico_max_size]
        all_pairs = all_pairs[:params.dico_max_size]

    # min dico size
    # Chris: temporarily commented while hacking
    #diff = all_scores[:, 0] - all_scores[:, 1]
    #if params.dico_min_size > 0:
    #    diff[:params.dico_min_size] = 1e9

    # confidence threshold
    if params.dico_threshold > 0:
        mask = all_scores > params.dico_threshold
        #mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs[mask.nonzero().squeeze()]
        logger.info("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))

    return all_pairs


def build_dictionary(src_emb, tgt_emb, params, s2t_candidates=None, t2s_candidates=None, append=False):
    """
    Build a training dictionary given current embeddings / mapping.
    """
    logger.info("Building the train dictionary ...")
    s2t = 'S2T' in params.dico_build
    t2s = 'T2S' in params.dico_build
    assert s2t or t2s

    if s2t:
        if s2t_candidates is None:
            s2t_candidates = get_candidates(src_emb, tgt_emb, params)
    if t2s:
        if t2s_candidates is None:
            t2s_candidates = get_candidates(tgt_emb, src_emb, params)
        t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    if params.dico_build == 'S2T':
        dico = s2t_candidates
    elif params.dico_build == 'T2S':
        dico = t2s_candidates
    else:
        s2t_candidates = set([(a, b) for a, b in s2t_candidates])
        t2s_candidates = set([(a, b) for a, b in t2s_candidates])
        if params.dico_build == 'S2T|T2S':
            final_pairs = s2t_candidates | t2s_candidates
        else:
            assert params.dico_build == 'S2T&T2S'
            final_pairs = s2t_candidates & t2s_candidates
            if len(final_pairs) == 0:
                logger.warning("Empty intersection ...")
                return None
        dico = torch.LongTensor(list([[a, b] for (a, b) in final_pairs]))

    logger.info('New train dictionary of %i pairs.' % dico.size(0))
    return dico.cuda() if params.cuda else dico
