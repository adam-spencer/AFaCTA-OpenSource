# AFaCTA-OpenSource
Fork of AFaCTA using [Ollama](https://ollama.com/) for selecting from a plethora of open source models instead of OpenAI models.

AFaCTA is an automatic annotation framework for Factual Claim Detection, focusing on verifiability and calibrating with self-consistency ensemble.

**This file requires editing and is not final 16/08/25.**

## How to use
### How to annotate a political speech?
```shell
python code/afacta_multi_step_annotation.py --file_name data/original-data/raw_speeches/AK1995_processed.csv --output_name AK1995 --context 1 --llm_name llama3.1:8b
```
In this example command, we annotate AK1995 with a context length of 1 (previous and subsequent sentence), using Llama3.1-8B.

### How to annotate a tweet dataset?
```shell
python code/afacta_twitter.py --file_name data/twitter/CT2022-1B.csv --llm_name llama3.1:8b
```
