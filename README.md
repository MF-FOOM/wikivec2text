# wikivec2text

Simple embedding -> text model trained on a tiny subset of Wikipedia sentences ([~7m](https://www.kaggle.com/datasets/mikeortman/wikipedia-sentences)) embedded via text-embedding-ada-002. 

Used to demonstrate the arithmetic properties of sentence embeddings. 

> **Warning**
> The [checkpoint on HF](https://huggingface.co/MF-FOOM/wikivec2text) is a `gpt2-small` finetuned heavily on a small subset of well-formatted Wikipedia sentences like `Ten years later, Merrill Lynch merged with E. A. Pierce.`
>
> Since these sentences are so structured and formal, it's very easy to go OOD if you're not careful. Stick to informational sentences that could plausibly appear in an encyclopedia, always start sentences with a capital letter, end with a period, etc.

## Acknowledgements

Built on [nanoGPT](https://github.com/karpathy/nanoGPT).
