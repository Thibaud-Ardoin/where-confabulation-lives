# How to install

Get the Llama3 model in ´Meta-Llama-3-8B´ folder. Should contain checklist.chk, consolidated.00.pth, params.json, tokenizer.model


# How to run

For a simple test run of the model.


```bash
torchrun --nproc_per_node 1 model_test.py
```

For a run that generates a dataset of activation according to a .yaml file:

```bash
torchrun --nproc_per_node 1 forward_pass.py
```