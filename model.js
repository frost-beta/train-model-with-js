import {core as mx, nn} from '@frost-beta/mlx'

// A decoder-only Transformer.
export default class Model extends nn.Module {
  constructor({vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads}) {
    super()
    this.embedding = new nn.Embedding(vocabSize, hiddenSize)
    this.pe = new nn.SinusoidalPositionalEncoding(hiddenSize)
    this.transformer = new nn.TransformerEncoder(numHiddenLayers,
                                                 hiddenSize,
                                                 numAttentionHeads)
    this.outProj = new nn.Linear(hiddenSize, vocabSize)
  }

  forward(x) {
    const L = x.shape[1]
    const mask = nn.MultiHeadAttention.createAdditiveCausalMask(L)
    x = this.embedding.forward(x)
    x = mx.add(x, this.pe.forward(mx.arange(L)))
    x = this.transformer.forward(x, mask)
    return this.outProj.forward(x)
  }
}
