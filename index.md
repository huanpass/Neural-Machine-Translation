## Abstract

Machine translation (MT) is an important approach in NLP domain with lots of commercial value and influence on fundamental theory of NLP. Considering the complexity of language models and difficulties in evaluating the results quantitatively, it is a challenge to get high quality translation result. After the initial review, it is found that the neural machine translation[1] method can address the challenge well. The first stage of the project is mainly about paper review regarding embedding features of words, sequence to sequence model[2], attention mechanism and BLEU evaluation method. The 2nd stage of the project aims to optimize an existing Neural-MT project via replacing part of the code (Encoder), importing the BLEU function and modifying the parameters (Optimizer/Layer type/dimension) to analysis the effects plus expanding the parsing section to support other language rather than just French. 


### Principles in Neural Machine Translation 

To represent a sentence in tensor form, it can be thought as the collection of tensors of each word. For the tensor of each word, one hot encoding can be too sparse and lack relevance between different words. A more efficient encoding method is to use the embedding feature of words. In this project, the initial one hot encoding was taken to convert word to index, then use word embedding layer from pytorch to convert index to tensor. The order of tensors of word in tensor of sentence reflects the order of words in the original sentence to maintain the semantics in structure. 
The sequence to sequence model is based on the conception that if we merge multiple languages in single network together, the differences can cause the network fails unless the network is quite large capable to find some high level pattern to co-exist different languages which can be difficult to converge and not efficient. To cope with this problem, sequence to sequence model was put forward to cooperate between 2 networks to utilize the information of each language maximally and efficiently while not distort each other. The encoder takes the input word one by one generating the context tensor in the end. Then, the decoder utilizes the <SOS> as the first input and context tensor from encoder as the first hidden state input to start its own loop. 
The secrete sauce is the attention mechanism. Normally, it is assumed that there exist strong relationship between the words in similar locations/structures of sentences. Rather than just provide a full context tensor of the encoder, an attention matrix was generated from encoder output to make the decoder to focus on certain locations in the sentence when translating certain locations. 


### The overview of the project
![where](https://lh3.googleusercontent.com/-fR3DLfVIFTE/WyD1tV-tQwI/AAAAAAAAANU/WmyizXNpIz8lvycLbuJOsa1iDPtESi71wCL0BGAs/w663-d-h802-n-rw/Untitled%2BDiagram%2B%25282%2529.png)


### Optimizatins 
Experiment setup: 50K training samples with 2K training samples as one epoch, 25 epochs in total. 
The training data set is got via randomly choosing 50K sentences  from a smaller data set (around 10k sentences ) converted from the original data set(150k sentences) via limiting the length of sentence within 10 words and starting with prefix like  "i am ", "i m ", "he is" etc. The evaluation data set is randomly 25K sentences chosen from the same smaller data set (around 10k sentences). 
Loss function: negative log likelihood loss 
Evaluation setting: BLEU (1K sentences for one epoch) and manual analysis (10 sentences in the end)

### Optimizer Adam Vs SGD
Data format: Starting time | Left time |#trained  %| Loss | BLEU value
![hoASFAw](https://lh3.googleusercontent.com/-6zbPJam5FLo/WyDtyuSTO6I/AAAAAAAAALE/wTKZXGrhv0A-PeSGJK0m-y_ruTJ98ISXQCL0BGAs/w663-d-h364-n-rw/SGDADAM.PNG)
![whe](https://lh3.googleusercontent.com/-H0TiBwPpfn0/WyA1n_z-zPI/AAAAAAAAAGc/4cEr6MZoKKozKhezJIdEjrofFKtLe2gwwCL0BGAs/w663-d-h277-n-rw/adam.PNG)

The experiment indicates that Adam converges faster in the initial period while then settle in a bad solution for a while before approaching the optimum solution. Through the final solution is better than SGD, the training time is 54% slower than SGD for the whole training set. The SGD through slow in the initial period, the improvement process is persistent and faster than the Adam Optimizer in the end. According to reviews to paper [3], it is found that adaptivity is more likely to overfit making it unable to find the optimal solution and generally worse than the solution of SGD though very popular. The experiment did not this phenomenon due to limited training speed. But, there is trace that Adam more likely to stay at the sub-optimum solution. 
Another issue observed from the training result is that even the loss is decreasing, the BLEU value can be worser indicating the loss function maybe not that good. I am thinking probably it will be better to use 1/(BLEU) as the loss function to guide the training for MT. 


### Decoder Implementation 
![hoASDAAyw](https://lh3.googleusercontent.com/-JTiACwglR1M/WyDy8HZ3cMI/AAAAAAAAAME/Lsu8w1K5FW0ywRbA_m5NqAUk7oWrh52RACL0BGAs/w663-d-h365-n-rw/yyyyyyyyyyyyy.PNG)

![how](https://lh3.googleusercontent.com/-rghISEoADVA/WyA1rbqxwNI/AAAAAAAAAGw/8k3WKRgZ01UnntFQdVknnGGFUoeGYZ_uwCL0BGAs/w663-d-h285-n-rw/decoder.PNG)

```markdown
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = self.dropout(output)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
        
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

```

The main motivation is to see whether such a complex decoder integrated with attention mechanism is necessary or not. I have implemented a simple decoder with only half of the original decoder based on one GRU layer, with embedding and dropout functions added. The result indicates without attention mechanism, the training time can be speedup by 20% with similar even better translation quality. 


### BLEU Importing 
```markdown
import nltk

def evaluatebleu(encoder, decoder, n=1000):
    BLEUscore = 0
    for i in range(n):
        pair = random.choice(pairs)
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        #print("WASP",output_sentence.strip().split())
        BLEUscore += nltk.translate.bleu_score.sentence_bleu([pair[1].strip().split()], output_sentence.strip().split())

    return BLEUscore/100

```
BLEU is a method based on n-gram model of sentences plus brevity penalty factor to evaluate the quality of translation via comparing the machine translation with human translation[4]. Basically, the loss function maybe too naïve for analyzing the differences in semantic information. So I imported one in this project by using a NLTK package. The final results displayed is the average of 1000 samples. 

###  Dimension of hidden layer 
128 vs 256 vs 512 

![hoASDAAw](https://lh3.googleusercontent.com/-aLOAjj3OHhc/WyDtgaT6DYI/AAAAAAAAAKY/sW0wp3SYQp41y4tvFPp9_b5DxkRe1T4OwCL0BGAs/w663-d-h270-n-rw/DIMENSSSSSSSSSSSS.PNG)

According to the experiment result, it is found that higher dimension for hidden layer can accelerate the training speed with more improvement with same size training set. One interesting found is the training quality is proportional to the training time. I think this is because the training is not complete or far from overfit. It can be an acceptable way to accelerate the training process with the cost of less accuracy. For the 512 dimension case, the training is too slow to finish it. 

### Parsing for Chinese Language
The index model of every word is generated via counting the words show up in data set and corresponding utf-8 code. And the sentence is just represented by the tensor based on these indexes. So, the neural machine translation model in this project does not need any prior knowledge about the type of language. It is in nature can deal with any language even something not language. What I have done is just removing some parsing requirement like capitalization, special symbols etc in the parsing section to make it also suitable for Chinese language. The result shows the potential that the model can actually be used for any language. 

### Conclusion
The project has reviewed several papers and projects regarding machine translation and general NLP issues. The efficiency of neural machine translation model has been verified. And SGD optimizer with default setting fails Adam with default setting in the final translation quality through faster in convergence in total. The review result also indicates Adam probably not good for some cases. As for the attention mechanism, my result indicates this technique maybe not more optimum than simple basic decode model while much more complex. It is not clear whether this conclusion still hold for larger size data set and more training epochs. The BLEU function has been imported with efficiency verified. As for the dimension of hidden layer, my result indicates higher dimension means more efficient utilization of data set while the training is much slower considering in my case the parameters have almost been doubled. Finally, the parsing support for Chinese language is done demonstrating the generality of the neural machine translation model. 

### Translation samples 
![h](https://lh3.googleusercontent.com/-C4D9CgLZT18/WyA44n4TvAI/AAAAAAAAAHo/0cidm4k10D88-F_FPXaS8-e8n6jqQ_wFwCL0BGAs/w663-d-h452-n-rw/ABC.PNG)

French - English                                   Chinese - English

### Attention mechanisim 
![hrr](https://lh3.googleusercontent.com/-9etlTBhjPak/WyCidqf5F-I/AAAAAAAAAIk/t1k-1xjA_gIuHNrqM7cobizaGt2RHhFfwCJoC/w663-h290-n-rw/111111.PNG)

### Reflection 
The training normally takes quite long time (2-3 hours) due to Pytorch does not support well for cloud platform. Should try with Tensorflow next time. 


### References
[1]https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

[2]Cho, Kyunghyun, Bart van Merrienboer, Çaglar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk and Yoshua Bengio. “Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.” EMNLP (2014).

[3]Wilson, A.C., Roelofs, R., Stern, M., Srebro, N., & Recht, B. (2017). The Marginal Value of Adaptive Gradient Methods in Machine Learning. NIPS.

[4]Papineni, K., Roukos, S., Ward, T., & Zhu, W. (2002). Bleu: a Method for Automatic Evaluation of Machine Translation. ACL.

                                               Auther: Zequan Zhou   Group 43   EECS349  zequanzhou2019 AT U DOT Northwestern DOT edu
