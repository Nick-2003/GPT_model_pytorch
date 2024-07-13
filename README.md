# GPT Model PyTorch

Implements a GPT Model in PyTorch on the Wine Reviews Dataset: https://www.kaggle.com/datasets/zynicide/wine-reviews

Further information can be found in the following blog post:

https://nathanbaileyw.medium.com/implementing-a-gpt-model-in-pytorch-15fd3f5d77ed

### Code:
The main code is located in the following files:
* main.py - Main entry file for training the network
* model.py - Implements the GPT model
* model_building_blocks.py - Embedding block, transformer block and causal attention mask implementations to use in the model
* dataset.py - Creates the Wine Review dataset
* train.py - Trains the PyTorch Model
* lint.sh - runs linters on the code
