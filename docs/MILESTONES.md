# Milestones

Numbered by what becomes *true*, not by how much work it was.

| # | State | Done |
| --- | --- | --- |
| 1 | The checkpoint can be opened, addressed and validated | ✅ |
| 2 | Model geometry and weight schema are typed and shape-checked | ✅ |
| 3 | The PyTorch oracle is captured and its format is read by tests | ✅ |
| 4 | Text → symbol ids matches the reference tokenizer exactly | ✅ |
| 5 | The text encoder matches the oracle within tolerance | ✅ |
| 6 | The stochastic duration predictor matches on fixed noise | |
| 7 | The flow, reversed, matches the oracle | |
| 8 | The HiFi-GAN decoder matches the oracle | |
| 9 | End-to-end synthesis on CPU matches the reference waveform | |
| 10 | The CLI synthesises a WAV from Tâi-lô/POJ on the command line | |
| 11 | CUDA kernels match the CPU reference, per kernel | |
| 12 | End-to-end CUDA synthesis is faster than PyTorch, measured | |

Milestone 9 is the one that matters. Everything before it is scaffolding, and
everything after it is optimisation — which cannot start until there is a
correct implementation to be faster *than*.

## What the numbering does not cover

Batching, streaming synthesis, and serving the result over HTTP are all
deliberately absent. They are answers to questions this project has not asked
yet, and the pipeline upstream already has an HTTP surface that works.
