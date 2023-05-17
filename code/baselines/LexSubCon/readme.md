We use the code from [LexSubCon](https://github.com/gmichalo/LexSubCon). 

We treat every word in the sentence as target, and use LexSubCon to give substitutions for every word. 
If the first substitution is not the same with original word, we regard this as an improvable target.

# Predicting

1. convert sws_test.json to Lexical Substitution format with `python convert2LS.py`
2. clone [LexSubCon](https://github.com/gmichalo/LexSubCon)
2. update LexSubCon/metrics/evaluation.py with updates/evaluation.py (we only revise the `write_results_lex_oot` function to only write improvable targets' substitutions)
3. using LexSubCon to directly predict substitution results (without any fine-tuning) `python main_lexical.py -tt test_LS.txt`
4. convert predicted oot result to SWS evaluation format with `python convert2SWS.py`