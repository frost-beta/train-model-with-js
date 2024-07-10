#!/usr/bin/env node

import fs from 'node:fs'
import path from 'node:path'
import {core as mx, optimizers as optim, nn, utils} from '@frost-beta/mlx'

import Model from './model.js'

// Hyperparameters.
export const contextSize = 128
export const hiddenSize = 128
export const numHiddenLayers = 8
export const numAttentionHeads = 4

// Traning configs.
const batchSize = 32
const epochs = 24
const learningRate = 1e-3

if (process.argv[1].endsWith('train.js'))
  main()

async function main() {
  // Use the text file for training.
  const filename = 'input.txt'
  const text = fs.readFileSync(filename).toString()

  // Create tokenizer.
  const vocab = getVocabulary(text)
  const {vocabSize, encode, decode} = getTokenizer(vocab)
  console.log('Vocabulary size is', vocabSize)

  // Encode the text to tokens, and split them for traning and validating.
  const data = encode(text)
  const dataTrain = data.slice(0, Math.floor(data.length * 0.9))
  const dataValid = data.slice(dataTrain.length)

  // Convert the tokens into features and labels.
  const {x: xTrain, y: yTrain} = loadData(dataTrain, contextSize)
  const {x: xValid, y: yValid} = loadData(dataValid, contextSize)
  console.log('Traning dataset\'s shape is', xTrain.shape)
  console.log('Validating dataset\'s shape is', xValid.shape)

  // Create model.
  const model = new Model({vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads})

  // Calculate how many parameters the model has.
  let nparams = 0
  for (const [k, x] of utils.treeFlatten(model.parameters())) {
    if (!k.includes('embedding'))
      nparams += x.size
  }
  console.log(`Model has ${(nparams / 1024 ** 2).toFixed(1)}M parameters.`)

  // Preprare utils for doing gradient descent.
  const lossAndGradFunction = nn.valueAndGrad(model, lossFunction)
  const optimizer = new optim.AdamW(learningRate)

  const reportPerIter = 100
  const totalIterations = epochs * Math.floor(xTrain.shape[0] / batchSize)

  // Train the model with training datasets.
  let losses = []
  for (let e = 0, iterations = 1, start = Date.now(); e < epochs; ++e) {
    for await (const [x, y] of iterateBatches(xTrain, yTrain, batchSize)) {
      // Use mx.tidy to free all the intermediate tensors immediately.
      mx.tidy(() => {
        const [loss, grads] = lossAndGradFunction(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
        losses.push(loss.item())
        // Keep the states of model and optimizer from getting freed.
        return [model.state, optimizer.state]
      })
      mx.dispose([x, y])
      // Report updates.
      if (++iterations % reportPerIter === 0) {
        const stop = Date.now()
        const trainLoss = mean(losses)
        console.log(`Iter ${iterations} (${(iterations / totalIterations * 100).toFixed(1)}%):`,
                    `Train loss ${trainLoss.toFixed(3)},`,
                    `It/sec ${(reportPerIter / (stop - start) * 1000).toFixed(3)}.`)
        start = Date.now()
        losses = []
      }
    }
  }
  const trainLoss = mean(losses)

  // Evaluate the model by running it with the validation dataset.
  model.eval()
  losses = []
  for await (const [x, y] of iterateBatches(xValid, yValid, batchSize)) {
    mx.tidy(() => {
      const loss = lossFunction(model, x, y)
      losses.push(loss.item())
      return [model.state, optimizer.state]
    })
  }
  const validLoss = mean(losses)
  console.log('Train Loss:', trainLoss.toFixed(3),
              'Valid Loss:', validLoss.toFixed(3))

  console.log('Save weights to weights.safetensors')
  model.saveWeights('weights.safetensors')
  console.log('Save tokenizer to tokenizer.json')
  fs.writeFileSync('tokenizer.json', JSON.stringify({vocab}))
}

// Analyze the text into a vocabulary.
function getVocabulary(text) {
  return Array.from(new Set(text.split(''))).sort()
}

// Create a simple character mapped tokenizer.
export function getTokenizer(vocab) {
  const vocabSize = vocab.length

  const itos = {}
  const stoi = {}
  vocab.forEach((c, i) => {
    itos[i] = c
    stoi[c] = i
  })

  const encode = (x) => x.split('').map(c => stoi[c])
  const decode = (x) => x.map(i => itos[i]).join('')

  return {vocabSize, encode, decode}
}

// Take tokens and split them into features and labels.
function loadData(tokens, contextSize) {
  let x = []
  let y = []
  for (let i = 0; i < tokens.length - contextSize - 1; i += contextSize) {
    x.push(tokens.slice(i, i + contextSize))
    y.push(tokens.slice(i + 1, i + contextSize + 1))
  }
  return {x: mx.array(x, mx.uint32), y: mx.array(y, mx.uint32)}
}

// Iterate the dataset in batches.
async function* iterateBatches(x, y, batchSize) {
  for (let i = 0; i < x.shape[0]; i += batchSize) {
    const slice = mx.Slice(i, i + batchSize)
    yield await new Promise((resolve) => {
      process.nextTick(() => resolve([x.index(slice), y.index(slice)]))
    })
  }
}

// Calculate the loss by 1) running the model with the inputs, and 2) then using
// cross entropy function to get the loss between the results and targets.
function lossFunction(model, x, y) {
  const logits = model.forward(x)
  const losses = nn.losses.crossEntropy(logits, y)
  return mx.mean(losses)
}

// Compute the mean value of an array.
function mean(array) {
  if (array.length == 0)
    return 0
  return array.reduce((a, b) => a + b) / array.length
}
