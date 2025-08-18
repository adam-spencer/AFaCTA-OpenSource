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

## References

The data used in this repository does not belong to me, it was accessed in accordance to the respective licenses from:
* Gupta, S., Singh, P., Sundriyal, M., Akhtar, M.S. and Chakraborty, T., 2021. Lesa: Linguistic encapsulation and semantic amalgamation based generalised claim detection from online content. arXiv preprint arXiv:2101.11891.
* Alam, F., Struß, J.M., Mandl, T., Mıguez, R., Caselli, T., Kutlu, M., Zaghouani, W., Li10, C., Shaar11, S., Shahi12, G.K. and Mubarak, H., The CLEF-2022 CheckThat! Lab on Fighting the COVID-19 Infodemic and Fake News Detection.
* Shaar, S., Alam, F., Martino, G.D.S., Nikolov, A., Zaghouani, W., Nakov, P. and Feldman, A., 2021. Findings of the NLP4IF-2021 shared tasks on fighting the COVID-19 infodemic and censorship detection. arXiv preprint arXiv:2109.12986.
* Dutta, S., Dhar, R., Guha, P., Murmu, A. and Das, D., 2022, December. A multilingual dataset for identification of factual claims in indian twitter. In Proceedings of the 14th Annual Meeting of the Forum for Information Retrieval Evaluation (pp. 88-92).
* Barrón-Cedeño, A., Alam, F., Caselli, T., Da San Martino, G., Elsayed, T., Galassi, A., Haouari, F., Ruggeri, F., Struß, J.M., Nandi, R.N. and Cheema, G.S., 2023, March. The clef-2023 checkthat! lab: Checkworthiness, subjectivity, political bias, factuality, and authority. In European conference on information retrieval (pp. 506-517). Cham: Springer Nature Switzerland.
* Ni, J., Shi, M., Stammbach, D., Sachan, M., Ash, E. and Leippold, M., 2024. Afacta: Assisting the annotation of factual claim detection with reliable llm annotators. arXiv preprint arXiv:2402.11073.
