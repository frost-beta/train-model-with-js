# Train text generation model with JavaScript

This repo hosts some educational scripts for doing basic training on a
decoder-only transformer, using [node-mlx](https://github.com/frost-beta/node-mlx)
with Node.js.

Files:

* `model.js` - defines the model.
* `input.txt` - text file used for training the model.
* `train.js` - script for traning.
* `generate.js` - script for generating text using the trained model.

## Platform

Only Macs with Apple Silicon are supported.

## How to use

Download dependencies and run the training script, which generates
`tokenizer.json` and `weights.safetensors`:

```bash
npm install
node train.js
```

Then use the generate script to actually generate some text from the weights:

```bash
node generate.js
```

## License

Public domain.
