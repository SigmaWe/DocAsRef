import functools
import evaluate
import numpy as np
import bert_score

def _compute(
    predictions,
    references,
    lang=None,
    model_type=None,
    num_layers=None,
    verbose=False,
    idf=False,
    device=None,
    batch_size=64,
    nthreads=4,
    all_layers=False,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False,
):
    if isinstance(references[0], str):
        references = [[ref] for ref in references]
    if idf:
        idf_sents = [r for ref in references for r in ref]
    else:
        idf_sents = None
    scorer = bert_score.BERTScorer
    if model_type is None:
        if lang is None:
            raise ValueError(
                "Either 'lang' (e.g. 'en') or 'model_type' (e.g. 'microsoft/deberta-xlarge-mnli')"
                " must be specified"
            )
        model_type = bert_score.utils.lang2model[lang.lower()]
    if num_layers is None:
        num_layers = bert_score.utils.model2layers[model_type]
    cached_bertscorer = scorer(
          model_type=model_type,
          num_layers=num_layers,
          batch_size=batch_size,
          nthreads=nthreads,
          all_layers=all_layers,
          idf=idf,
          idf_sents=idf_sents,
          device=device,
          lang=lang,
          rescale_with_baseline=rescale_with_baseline,
          baseline_path=baseline_path,
          use_fast_tokenizer=use_fast_tokenizer,
    )
    (P, R, F) = cached_bertscorer.score(
        cands=predictions,
        refs=references,
        verbose=verbose,
        batch_size=batch_size,
    )
    output_dict = {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F.tolist(),
    }
    return output_dict



bertscore = evaluate.load("bertscore")

# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi"]

cands = np.array(["paul merson was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . andros townsend scored the tottenham midfielder in the 89th minute . paul merson had another dig at andros townsend after his appearance . the midfielder had been brought on to the england squad last week . click here for all the latest arsenal news news .",
       "paul merson has restarted his row with andros townsend . the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . andros townsend scores england 's equaliser in their 1-1 friendly draw with italy in turin .",
       "paul merson has restarted his row with andros townsend after the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley on sunday . townsend was brought on in the 83rd minute for tottenham as they drew 0-0 against burnley . townsend hit back at merson on twitter after scoring for england against italy .",
       "paul merson has restarted his row with andros townsend . the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . townsend was brought on in the 83rd minute for tottenham as they drew 0-0 with burnley .",
       "paul merson has restarted his row with andros townsend after the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley on sunday . merson initially angered townsend for writing in his sky sports column that ` if andros townsend can get in -lrb- the england team -rrb- then it opens it up to anybody ' townsend was brought on in the 83rd minute for tottenham as they drew 0-0 against burnley .",
       "paul merson has restarted his row with andros townsend after the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . merson initially angered townsend for writing in his sky sports column that ` if andros townsend can get in -lrb- the england team -rrb- then it opens it up to anybody . ' paul merson had another dig at andros townsend after his appearance for tottenham against burnley .",
       "paul merson has restarted his row with andros townsend after the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0 - 0 draw with burnley on sunday . # rubberdub # 7minutes , ' merson put on twitter . merson initially angered townsend for writing in his sky sports column that ' if andros townsend can get in ( the england team ) then it opens it up to anybody . '",
       "paul merson criticised andros townsend's call-up to the england squad . townsend hit back at merson after scoring for england against italy . the tottenham midfielder was brought on in the 83rd minute against burnley .",
       "Paul Merson is not happy with Andros Townsend's call-up to the England squad last week",
       "paul merson had a dig at andros townsend after his appearance for tottenham . townsend was brought on in the 83rd minute for tottenham against burnley . 'just been watching the game, did you miss the coach? #rubberdub #7minutes,' merson wrote on twitter .",
       "paul merson has restarted his row with andros townsend after the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley on sunday . ' paul merson had another dig at andros townsend after his appearance for tottenham against burnley . townsend was brought on in the 83rd minute for tottenham as they drew 0-0 against burnley .",
       "paul merson has restarted his row with andros townsend after the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley on sunday . ` just been watching the game , did you miss the coach ? #rubberdub # 7minutes , ' merson put on twitter .",
       "Tottenham drew 0-0 with Burnley at Turf Moor on Sunday . Andros Townsend was brought on in the 83rd minute for Tottenham . Paul Merson criticised Townsend 's call-up to the England squad last week . Townsend hit back at Merson on Twitter after scoring for England against Italy .",
       "paul merson has restarted his row with andros townsend after the tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . townsend was brought on in the 83rd minute for tottenham as they drew 0-0 against burnley . paul merson had another dig at andros townsend after scoring for england against italy .",
       "paul merson criticised townsend 's call-up to the england squad last week . andros townsend was brought on in the 83rd minute for tottenham as they drew 0-0 against burnley on sunday . townsend hit back at merson after his appearance for england against italy .",
       "paul merson has restarted his row with burnley on sunday . townsend was brought on in the 83rd minute for tottenham . andros townsend scores england 's equaliser in their 1-1 friendly draw . townsend hit a stunning equaliser for england against italy .",
       "Andros Townsend scored for England against Italy on Wednesday . Paul Merson criticised Townsend's call-up to the England squad . Merson said Townsend should not have been in Roy Hodgson's squad . Townsend hit back at Merson on Twitter after scoring for England ."],
      dtype=object)

refs = np.array(["Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up.",
       "Andros Townsend an 83rd minute sub in Tottenham's draw with Burnley. He was unable to find a winner as the game ended without a goal. Townsend had clashed with Paul Merson last week over England call-up."],
      dtype=object)

# a = _compute(predictions=cands, references=refs, use_fast_tokenizer=True, idf=True, model_type="roberta-base")
a = bertscore.compute(predictions=cands, references=refs, use_fast_tokenizer=True, idf=True)

print(a)
