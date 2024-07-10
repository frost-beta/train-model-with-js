#!/usr/bin/env node

import fs from 'node:fs'
import nextTick from 'tick-promise'
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
  // Pass the tokens to the model and get the next token.
  const forward = (tokens) => {
    const inputs = mx.array([ tokens.slice(-contextSize) ], mx.int32)
    const logits = model.forward(inputs)
    return sample(logits.index(0, -1), temperature)
  }

  let tokens = prompt
  while (true) {
    const token = mx.tidy(() => forward(tokens).item())
    tokens.push(token)
    // Yield the result in the next tick of loop, so GC can get a chance to run.
    await nextTick()
    yield token
    // Stop when hit maxTokens limit.
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
