# Introduction

These are CNTK BrainScript examples solution, which has three projects.

## List of Projects

1. **AN4:**

This is an example for training feed forward and LSTM networks for speech data.
You only need to run script “FeedForward.cntk” or “LSTM-NDL_ndl_deprecated.cntk" for FF and LSTM training respectively.

The traing data contents of this directory is a modified version of AN4 dataset (The AN4 dataset is a part of CMU audio databases) pre-processed and optimized for CNTK end-to-end testing.

The data uses the format required by the HTKMLFReader. For details please refer to the documentation.


2. **CMUDict:**

This is an example demonstrates the use of CNTK for grapheme-to-phoneme (letter-to-sound) conversion using a sequence-to-sequence model with attention, using the CMUDict dictionary.

3. **MNIST:**

This is an example demonstrates usage of the NDL (Network Description Language) to define networks, using MNIST dataset.

# How to Run

1. Open the "CNTKBrainscriptExamples.sln" solution. (It will open with Visual Studio 2017 by default.)

2. Set the project you want as "Startup Project"

3. Set the script you want to run as "Startup File"

4. Click “Run CNTK Brain Script”
