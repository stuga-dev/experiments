# MIRACL

MIRACL [1] is a multilingual retrieval dataset over Wikipedia. Native speakers wrote each question
and judged which passages answer it.

This set uses MMTEB's reranking version of the dev split, [mteb/MIRACLReranking](https://huggingface.co/datasets/mteb/MIRACLReranking)
at revision `ab6f54eff185a84bc1f6ab96b56bc7df87433228`. It gives each question a first-stage list of
100 candidate passages, with MIRACL's judgments as its qrels.

`scripts/build-miracl.ts` builds the set as follows:

- **Languages:** English, Chinese, Japanese, Korean, Arabic and Thai, the languages of Stuga's
  privacy-laws sample.
- **Candidates:** the first 24 of each list, in the list's order. That order is the "no reranker"
  baseline.
- **Questions:** 40 per language, drawn with a fixed seed from the questions that have a relevant
  passage among their first 24. `eligibility.json` gives, per language, the dev questions and how many
  are eligible.

## Licence

- Passage text is from Wikipedia, under CC BY-SA 4.0; `corpus.jsonl` carries the same licence.
- MIRACL's topics and judgments are Apache-2.0.
- MMTEB publishes the reranking version under CC BY-SA 4.0.

## Caveat

The reranking version marks every candidate that is not a judged answer as not relevant. A relevant
passage that no annotator saw therefore counts as wrong, for every ranker alike.

[1] X. Zhang, N. Thakur, O. Ogundepo, E. Kamalloo, D. Alfonso-Hermelo, X. Li, Q. Liu,
M. Rezagholizadeh, J. Lin. MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages.
TACL 11, 2023.
