## Abstract

Machine translation (MT) is an important approach in NLP domain with lots of market value and influence on fundamental theory of NLP. Considering the complexity of language models and difficulties in evaluating the results quantitatively, some novel models and evaluation method are required to achieve meaningful results. After the initial review, it is found that the neural machine translation method can address the challenge well. So, the first stage of the project is mainly about paper review regarding embedding features of words, sequence to sequence model, attention mechanism and BLEU evaluation method. The 2nd stage of the project aims to optimize an existing Neural-MT project via replacing part of the code (Encoder), importing the BLEU function and modifying the parameters (Optimizer/Layer type/dimension) to analysis the effects plus expanding the parsing section to support Chinese font. 

### Principles in Neural Machine Translation 

To represent a sentence in tensor form, it can be thought as the collection of tensors of each word. For the tensor of each word, one hot encoding can be too sparse and lack relevance between different words. A more efficient encoding method is to use the embedding feature of words. In the project, the initial one hot encoding was taken to convert word to index, then use word embedding layer from pytorch to convert index to tensor. The order of tensors of word in tensor of sentence reflects the order of words in the original sentence to maintain the semantics in structure. 
The sequence to sequence model is based on the conception that if we merge multiple languages in single network together, the differences can cause the network fails unless the network is quite large capable to find some high level pattern to co-exist different languages which can be difficult to converge and not efficient. So, sequence to sequence model aims to cooperate between 2 networks to utilize the information of each language maximally and efficiently while not distort each other. The encoder takes the input word one by one generating the context tensor in the end. Then the decoder utilizes the <SOS> as the first input and context tensor from encoder as the first hidden state input to start its own loop. 
The secrete sauce is the attention mechanism. Normally, it is assumed that there exist strong relationship between the words in similar locations/structures of sentences. Rather than just provide a full context tensor of the encoder, an attention matrix was generated from encoder output to make the decoder to focus on certain locations in the sentence when translating certain locations. 

### The big picture 
![where](https://lh3.googleusercontent.com/-U5VE5GRIFdU/WyAvmP2FV0I/AAAAAAAAAFs/VDr2A_FTU_ADx_v1tIW9IlK4VVSdttIXwCL0BGAs/w663-d-h1009-n-rw/Untitled%2BDiagram%2B%25281%2529.png)

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block


```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).


### Optimizatins 
Experiment setup: 50K training samples with 2K training samples as one epoch, 25 epochs in total. 
The training data set is got via randomly choosing 50K sentences  from a smaller data set (around 10k sentences ) converted from the original data set(150k sentences) via limiting the length of sentence within 10 words and starting with prefix like  "i am ", "i m ", "he is" etc. The evaluation data set is randomly 25K sentences chosen from the same smaller data set (around 10k sentences). 
Loss function: negative log likelihood loss 
Evaluation setting: BLEU (1K sentences for one epoch) and manual analysis (10 sentences in the end)
Optimizer 
### Adam Vs SGD
![whe](https://lh3.googleusercontent.com/-H0TiBwPpfn0/WyA1n_z-zPI/AAAAAAAAAGc/4cEr6MZoKKozKhezJIdEjrofFKtLe2gwwCL0BGAs/w663-d-h277-n-rw/adam.PNG)
The experiment indicates that Adam converges faster in the initial period while then settle in a bad solution for a while without further improvement. The SGD through slow in the initial period, capable of finding a better solution gradually in the end. According to reviews to papers [], it is found that adaptivit is more likely to overfit making it unable to find the optimal solution and generally worser than the solution of SGD though very popular. 

### Decoder Implementation 
![how](https://lh3.googleusercontent.com/-rghISEoADVA/WyA1rbqxwNI/AAAAAAAAAGw/8k3WKRgZ01UnntFQdVknnGGFUoeGYZ_uwCL0BGAs/w663-d-h285-n-rw/decoder.PNG)
The main motivation is to see whether such a complex decoder integrated with attention mechanism is necessary or not. I have implemented a simple decoder based on one GRU layer, with embedding and dropout functions added. The result indicates without attention mechanism, the training time can be speedup by 11.1% with similar even better translation quality. 


### BLEU Importing 
BLEU is a method based on n-gram model of sentences plus brevity penalty factor to evaluate the quality of translation via comparing the machine translation with human translation. Basically, the loss function maybe too naïve for analyzing the differences in semantic information. So I imported one in this project by using a NLTK package. The final results displayed is the average of 1000 samples. 
Dimension of hidden layer 
128 vs 256 vs 512 

### Parsing for Chinese Language
The index model of every word is generated via counting the words show up in data set and corresponding utf-8 code. And the sentence is just represented by the tensor based on these indexes. So, the neural machine translation model in this project does not need any prior knowledge about the type of language. It is in nature can deal with any language even something not language. What I have done is just removing some parsing requirement like capitalization, special symbols etc in the parsing section to make it also suitable for Chinese language. The result shows the potential that the model can actually be used for any language. 

### Conclusion
The project has reviewed several papers and projects regarding machine translation and general NLP issues. The efficiency of neural machine translation model has been verified. And SGD optimizer with default setting won Adam with default setting in the final translation quality through faster in convergence in the initial stage. The review result also indicates Adam actually bad for a broader range of cases. As for the attention mechanism, my result indicates this technique maybe not more optimum than simple basic decode model while much more complex. It is not clear whether this conclusion still hold for larger size data set and more training epochs. The BLEU function has been imported with efficiency verified. As for the dimension of hidden layer, my result indicates higher dimension means more efficient utilization of data set while the training is much slower considering in my case the parameters have almost been doubled. Finally, the parsing support for Chinese language is done demonstrating the generality of the neural machine translation model. 




                                                                                             Arther: Zequan Zhou   Group 43   EECS349
