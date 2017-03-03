# Lossless compression using Neural Networks

## 1. Overview

### Motivation

Arithematic encoding has been used since past 20 years to achieve close to entropy compression for known distributions. Adaptive variants of Arithematic encoding \(for a chosen k-context model\) have also been designed, which first try to learn the conditional k-th order distribution, in the first pass nad use the same for compression in the second pass.

However, as the complexity increases exponentially in $k$, with the alphabet size. Generally the context is limited to $k = 5,10$. Higher values of context are not tractable. Can we consider RNN based models to achieve improved conditional probability, which in turn can be used along with arithmatic encoding.

Another important motivation this serves it with respect to how well can RNN's learn the probability distributions for compression, which can in turn help in intuitive understanding of RNN based image/video compression \(lossless or lossy\).

### Past work

There has been a lot of work on sequence prediction using RNN's, where the aim is to generate a sequence which resembles a given dataset. For eg: generating shakespeare's plays, etc.

1. Unreasonable Effectiveness of RNN: [http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
2. LSTM based text prediction: [http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf](http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf)
3. Prediction using RNNs \[Graves et.al.\]: [https://arxiv.org/abs/1308.0850](https://arxiv.org/abs/1308.0850)

There was also some work in early 2000's on lossless compression using neural networks. However, due to Vanilla RNN's not being able to capture long term dependencies well, the models might not have performed as well. Also in the past 5 years, the speeds of neural network have dramatically improved, which is a good thing for NN based probability estimators.

1. Neural Networks based compression: [http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=478398](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=478398)
2. Matt Mahoney Implementation: [http://mattmahoney.net/dc/nn\_paper.html](http://mattmahoney.net/dc/nn_paper.html)

Also on the theoretical side, there are connections between predictors trained with log-loss and universal compression. Thus, if RNN's can act as good predictors, we should be able to utilize them into good compressors.n

1. EE376c Lecture Notes on Prediction: [http://web.stanford.edu/class/ee376c/lecturenotes/Chapter2\_CTW.pdf](http://web.stanford.edu/class/ee376c/lecturenotes/Chapter2_CTW.pdf)

Another interesting thing to note is that, RNN based models have been partially used in the state-of-the-art lossless compressors. They have been mainly used only for context mixing. The compressors find the probability of the next character based on multiple human-designed contexts/features \(eg: past 20 chars, 4 words, or alternate characters, only the higher bits of the last 10 bytes etc.\). These probabilites are "mixed" \(somethig like boosting using experts\), using a LSTM based context mixer.  
In fact, most of the leading text compressors, on the [Hutter prize](http://prize.hutter1.net/) leaderboard use LSTMs for model mixing. For example, here is the flowchart for the [CMIX](http://www.byronknoll.com/cmix.html) use LSTM's for context mixing. 

![cmix_image](http://www.byronknoll.com/images/architecture.png)

### Applications

1. **Improved intuitive understanding** of RNN based structures for compression. The understanding can be used later to make improvements to more complex image/video compressors
2. **Wide Applications** to generic text/DNA/parameter compression. i.e. wherever arithematic encoding is used.
3. **Theoretical Connections** with log-loss based predictors, can be understood based on simple linear-RNN networks etc. 

## 2. Experiments
We plan to conduct some fundamental experiments first before going on to compress real DNA/Text dataset. 

### IID sources
We first start with simplest sources, i.i.d sources over binary alphabet and see if we can compress them well. We can show that the expected cross entropy loss for i.i.d sequences has a lower bound of binary entropy of the soruce. Thus the aim is to read this log-loss limit, which will confirm that arithematic encoding will work well. 

We observe that for iid sources, even a small model like a [8 cell, 2 layer network] is able to perform optimally with a very small (1000 length training sequence). 

### 0-entropy sources
Our next sources are 0-entropy stationary sources. By 0-entropy we mean they have 0 entropy rate (the `$m^th$` order entropy converges to 0 as `$m \rightarrow \infty$`). 
Our sources are very simple binary sources such as: 

```mathjax
X_n = X_{n-1} + X_{n-k}
```
where k is the parameter we choose. (the + is over binary alphabets). In this case, we observe that the process is stationary and is deterministic once you fix the first `$k$` symbols. Thus, it has entropy rate 0. Note that it seems iid until order `$k-1$`. Thus, any sequence modelling it with a lower model wont be able to compress at all. 

We conduct experiment by varying `$k$`. Higher `$k$` are generally more difficult to compress for standard compressors like LZ (and in-fact a lower order adaptive arithematic encoder wont be able to compress at all). Some of the observations are as follows:

#### Parameters:
   * All results for 2 epoch runs \(1 epoch training & 1 epoch compression\)
   * The input files generated are of size `$10^8$`, which is also one of the standard lengths for compression comparison.
   * Model is a 32 cell 3 & 2 layer (we also try other models, but more on that later)
   * The sequence length of training was 64 \(lengths higher than 64 will get difficult to train\)
   * Unlike standard RNN models, we retain the state at the end of the batch, so that the state can be passed correctly to the next chunk. Also, the batches are parsed sequentially through the text.
   * We have a validation set, which we reun every 100th iteration. The validation text is also generated with the same parameters and is of length 10,000. 


#### 2-layer network
The graph shows the learning curve for the 2-layer model. The learning curves are for different inputs with markovity [10,20,30,40,50,60]. We observe that the model takes longer time to learn higher markovity models. This is expected as the model tries to explore every order (intuitively), and tries out smaller orders before going on to higher ones. 
Also, observe that the model is not able to learn at all in 1 epoch from markovity 60. 

![val-2](images/val_64_2_layer.png)

#### 3-layer network
The 3-layer model also has similar difficulties, as it is also not able to learn for markovity 60 text. This suggests that, this has to do with the information flow, and perhaps might be due to vanishing gradients issue. 

![val-1](images/val_64_3_layer.png)

The overall resutls are as follows. The numbers are bits per symbol. As the input is binary, the worst we can do should be 1 bits/symbol. We compare the results with a universal compressor XZ. 

| Markovity | 3-layer NN | 2-layer NN | XZ |
| --- | --- | --- | --- |
| 10 | 0.0001 | 0.0001 | 0.004 |
| 20 | 0.0002 | 0.0005 | 0.05 |
| 30 | 0.01 | 0.07 | 0.4 |
| 40 | 0.1 | 0.27 | 0.58 |
| 50 | 0.3 | 0.4 | 0.65 |
| 60 | 1 | 1 | 0.63 |

The results showcase that, even over pretty large files ~100MB, the models perform very well for markovity until 40-50. However, for longer markovity, it is not able to figure out much, while LZ figures out some things, mainly becasue of the structure of the LZ algorithm. \(I will regenerate data for markovity 60 a few more times to confirm, as 0.63 looks a bit low than expectations\).

This suggests that, we should be able to use LZ based features along with the NN to improve compression somehow. This also suggests that, directly dealing with sources with very long dependencies (images, DNA) in a naive way would not work due 50 markovity limit.

#### Analysis of how sequence length impacts the learning

It was observed that sequence length while we perform truncated backproagation dramatically impacts the learning. One positive is that, the network does learn dependencies longer than sequence length sometimes. Although very long sequence lengths will suffer from vanishing gradients issue, which we need to think how to solve.

For a 16-size 2 layer network, with sequence length of 8, we were able to train for markovity 10 very well (thus even though we do not explicitly backproagate, there is still some learning below 8 levels). However, anything above that  
\(markovity 15, 20, ...\) gets very difficult to train.

![train-1](images/loss_8.png\)
![val-1]\(images/val_loss_8.png)
4. Try compressing images: Eg: [https://arxiv.org/abs/1601.06759](https://arxiv.org/abs/1601.06759)

## Feb 17 Update

### IID sources 

I tried with some small markov sources and iid sources. The network is easly able to learn the distribution \(within a few iterations\).

### 0 entropy sources.

For 0 entropy sources such as:   
    X_n = X_{n-20} exor X\_{n-k}

For sequence lengths of 10^7, 10^8 we are able to capture dependence very well for sources with $k &lt; 20,22$ with relatively small RNN networks \(1024, 3-layer networks\)  
However, above that markovity I am finding it difficult to train the network. Also, sometimes the network fails for smaller values of $k$ as well.   
I am still not sure what the reason is, currently trying some techniques of training.

![k-training](char-rnn-tensorflow/images/img2.png)

### DNA Dataset

I tried with two real datasets, The first one is the chromosome 1 DNA dataset \(currently the model only supports 1D structures, so trying with sequences/text first\). For DNA compression, the LZ77 based compressors \(gzip etc. \) achieve 1.9 bits/base, while more state-of-the art custom compressors achieve 1.6 bits/base. Neural network based compressor achieved close to 1.6 bits/base compression. Which was encouraging.

### Hutter prize dataset

The Hutter prize is a competition for compressing the wikipedia knowledge dataset \(100MB\) into 16MB or less. Compressors like gzip are able to perform upto 34MB, while more carefully preprocessed LZTurbo, can perform upto 25MB. The best, state of the art compressors, \(which incidentally also use neural networks for context mixing\) perform close to 15MB. Our basic character-level model (1024 size 3 layer) performs close to 16.5MB compression, which again is comparatively good.

![hutter](char-rnn-tensorflow/images/img3.png)


## Feb 24 Update

We are able to train 0-entropy sources until a significantly high markovity in the first epoch itself. There are a few significant changes to the model to achieve this:

1. Retain the state from the previous batch. This is not the default behaviour in deep learning, as RNN's are generally applied on a single sentence during training. We explicitly store the state and reassign it. 
2. To get this working during training, we are as of now restricting BATCH\_SIZE=SEQ\_LENGTH\( SEQ\_LENGTH is the number of timeframes you backpropagate\)
3. We use simpler models \(32/64 sized 2-layer models, instead of 1024 3 layered models, as simpler models train quicker, and we dont need a bigger model for this application, which is also a good thing in itself, as it makes a lot of things better/faster\)
4. Lower learning rate helps for larger models for better convergence 
5. The baremetal code is here: [15\_char\_rnn\_gist.py](NN_compression/tf_char_rnn/15_char_rnn_gist.py)

### Improvements

We improve upon the previous results by training for Markovity 40 \(as against 20 in the previous case\). Experiments with higher markovity are ongoing.

Also, I kept the DNA compression and text compression code running, and both of them increased at a steady rate \(but slow\). The DAN dataset went from 1.5 bits/base  -&gt; 1.35 bits/base, and the Text dataset came to 16.5MB \(which is close to the 16MB competition limit, although excluding the decompressor\).

I believe, using simpler models, with the new changes can significantly boost the performance, which I am planning to do next.

### TODO

1. Check how well the models generalize
2. Run it on images/video? \(still needs some work\): see PixelRNN
3. Read more about the context mixing algorithms used in video codecs etc.




