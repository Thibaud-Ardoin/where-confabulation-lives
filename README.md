# How to install

Get the Llama3 model in ´Meta-Llama-3-8B´ folder. Should contain checklist.chk, consolidated.00.pth, params.json, tokenizer.model

To install the required packages, run:

```bash
pip install -r requirements.txt
```

# How to run

The configuration file is ´params.yaml´. To use the dvc pipeline controller, use:

```bash
dvc repro dvc.yaml
```

This will load the data files from text form until the generation of the detection models and the steering vectors.

With this Steering vector you might want to run the testing script that will alterate the generation process:

```bash
torchrun --nproc_per_node 1 model_test.py
```


For a simple forward pass that generates a dataset of activation according to a .yaml file:

```bash
torchrun --nproc_per_node 1 forward_pass.py
```