# steps to get distantly supervised SWS data

## Directly Download

https://mtutorresstorage.blob.core.windows.net/swsdata/wiki_170_s-lexical-1-1125.json


## Manually Generate

1. Go to [wikidata](https://dumps.wikimedia.org/enwiki/) and download several .xml.bz2 files (e.g. https://dumps.wikimedia.org/enwiki/20221020/enwiki-20221020-pages-articles-multistream1.xml-p1p41242.bz2, note that the url will be expired, please only refer to the format of the link). You can download as much files as you need, each file may contain around 00-200 million sentences. 
2. Use [wikiextractor](https://github.com/attardi/wikiextractor) to process those .xml.bz2 files. (e.g. `python -m wikiextractor.WikiExtractor enwiki-20221020-pages-articles-multistream2.xml-p1p41242.bz2 -o ./outwiki2 -b 5M --json --processes 5`)
3. Download PPDB files. [cite](https://aclanthology.org/N13-1092/) [download_page](http://paraphrase.org/#/download). We use ppdb-2.0-s-lexical, which is 'English' 'lexical' 's size' pack in the download page ([download_file_url](http://nlpgrid.seas.upenn.edu/PPDB/eng/ppdb-2.0-s-lexical.gz))
4. Get thesaurus from [webster](www.merriam-webster.com) using `python get_thesaurus.py`.
5. Generate distantly supervised data with `python get_sws_ds.py`

