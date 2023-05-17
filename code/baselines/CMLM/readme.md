We use the code from [Mask-Predict](https://github.com/facebookresearch/Mask-Predict). 

During training, we treat SWS as a translation task, the translation result is to replace all the improvable targets in the sentence. 
During Predicting, we let CMLM to "translate" the sentence, and we compare the predicted sentence with original one to get suggestions. 

# Training

1. run `python convert2CMLM.py` to convert SWS json files into CMLM files. 
2. clone [Mask-Predict](https://github.com/facebookresearch/Mask-Predict)
3. let {data_path} = code/baselines/CMLM/data, then
    ```
    python preprocess.py --source-lang en --target-lang de --trainpref {data_path}/train.en-de --validpref {data_path}/valid.en-de --testpref {data_path}/test.en-de --destdir sws/data-bin  --workers 60 --joined-dictionary

    python train.py sws/data-bin --arch bert_transformer_seq2seq --share-all-embeddings --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 --lr 5e-5 --warmup-init-lr 1e-8 --min-lr 1e-10 --lr-scheduler inverse_sqrt --warmup-updates 10000 --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_self --max-tokens 3072 --weight-decay 0.01 --dropout 0.3 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 --max-source-positions 10000 --max-target-positions 10000 --max-update 1000000 --seed 0 --save-dir sws/ckpt 2>&1 > log
    ```

# Predicting

1. update generate_cmlm.py in the repo with updates/generate_cmlm.py
2. run script
   ```
   python generate_cmlm.py sws/data-bin --path sws/ckpt/checkpoint_best.pt --task translation_self --remove-bpe --max-sentences 15 --decoding-iterations 10  --decoding-strategy mask_predict
   ```