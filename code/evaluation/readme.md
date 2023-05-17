usage 1: as a script

`python score.py --gold ../../data/sws/sws_test.json --pred {pred_json_path} --name {experiment_name}`

usage 2: as a python function

```py
from score import eval
res = eval('../../data/sws/sws_test.json', 'pred_json_path')
```