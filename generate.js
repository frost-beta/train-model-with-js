#!/usr/bin/env node

import fs from 'node:fs'
import {core as mx} from '@frost-beta/mlx'

import Model from './model.js'
import {contextSize, hiddenSize, numHiddenLayers, numAttentionHeads, getTokenizer} from './train.js'

main()

async function main() {
  // Create tokenizer.
  const {vocab} = JSON.parse(fs.readFileSync('tokenizer.json'))
  const {vocabSize, encode, decode} = getTokenizer(vocab)

  // Create model.
  const model = new Model({vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads})
  model.loadWeights('weights.safetensors')

  // Generate.
  const prompt = 'MIRANDA:\n'
  process.stdout.write(prompt)
  for await (const token of step(encode(prompt), model)) {
    const char = decode([token])
    process.stdout.write(char)
  }
  process.stdout.end('\n')
}

// Generate tokens from prompt.
async function* step(prompt, model, maxTokens = 512, temperature = 0.8) {
  const forward = (tokens) => {
    let logits = model.forward(mx.array([tokens.slice(-contextSize)], mx.int32))
    logits = logits.index(mx.Slice(), -1, mx.Slice())
    return sample(logits, temperature)
  }

  let tokens = prompt
  while (true) {
    const token = mx.tidy(() => forward(tokens).item())
    tokens.push(token)
    yield await new Promise(resolve => process.nextTick(() => resolve(token)))

    if (tokens.length > maxTokens)
      return
  }
}

// Pick the best token from logits.
function sample(logits, temperature) {
  if (temperature == 0)
    return mx.argmax(logits, -1)
  else
    return mx.random.categorical(mx.multiply(logits, 1 / temperature))
}
