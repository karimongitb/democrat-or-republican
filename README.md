# democrat-or-republican, made by karim
Basic DistilBERT model training code that predicts whether a statement is likely to be from a U.S. Democrat or a Republican.<br>
Uses the [Jacobvs/PoliticalTweets dataset](https://huggingface.co/datasets/Jacobvs/PoliticalTweets)<br>
python3, code is in train.py<br>
evaluation results with the basic fast settings fom train.py, highly recommended to change based on your device and goals

| Epoch | Training Loss | Validation Loss | Accuracy  | F1       |
|-------:|--------------:|----------------:|----------:|---------:|
| 1     | 0.214100      | 0.292491        | 0.868871  | 0.868871 |
| 2     | 0.140200      | 0.283668        | 0.880840  | 0.880838 |
| 3     | 0.095000      | 0.347880        | 0.882520  | 0.882520 |

## purpose
- aims to investigate whether a prompt can be distinguished as uniquely "Democratic" or "Republican"
- and thus analyze model behavior
- highlight the biases that could be encountered
- general research and understanding

## limitations & biases
- short answer: there are many.
- long answer:
  - not everyone in the Democratic or Republican party share the same views. so a generation/tweet cannot represent EVERYONE in that party
  - tweet could contain sarcasm or simple comments, affecting results
  - could over- or under-represent demographics and regions (distributional bias)
  - the most recent tweet in the dataset was posted on 2023-02-19 23:32:00, politics change and may not reflect the party/politician's current view
  - short tweets (and thus prompts) lack context and could produce seemingly random generations
  - specific messages but with keywords could potentially trigger a false-positive
  - etc...

## evaluations
- an incredibly basic "benchmark" or "evaluation", used part of, selecting 250 with seed 67, from the original test set in train.py to evaluate.
- this model was trained in a rather basic manner meant for speed, thus, its accuracy could potentially even be improved.
- some of the tweets could not be enough to identify the political party, e.g. short, vague messages that could also be filtered out
- there are many reasons why this is not exactly a good measure, such as sampling size and the fact that confidence tests have been omitted, etc. but it can still show something
- note that the models were supplied with a well-structured prompt specifically tailored to work with the benchmark.

| Model                  | Correct | Total | Accuracy (%) |
|-----------------------:|--------:|------:|-------------:|
| democrat-or-republican | 221      | 250    | **88.40**    |
| DeepSeek-V3.1          | 221      | 250    | **88.40**    |
| Qwen3-235B-A22B        | 221      | 250    | **88.40**    |
| gpt-oss-120b           | 210      | 250    | **84.00**    |
