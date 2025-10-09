
### Encoder
The Encoder **encodes the input image with 3 color channels into a smaller image with "learned" channels**.
This smaller encoded image is a summary representation of all that's useful in the original image.
i don't need to train an encoder from scratch. Why? Because there are already CNNs trained to represent images.
I chosen to use the **101 layered Residual Network trained on the ImageNet classification task**, already available in PyTorch. As stated earlier, this is an example of Transfer Learning. You have the option of fine-tuning it to improve performance.
![ResNet Encoder](https://raw.githubusercontent.com/0xprv/PyTorch-Image-to-Caption/refs/heads/main/encoder.png)
These models progressively create smaller and smaller representations of the original image, and each subsequent representation is more "learned", with a greater number of channels. The final encoding produced by our ResNet-101 encoder has a size of 14x14 with 2048 channels, i.e., a `2048, 14, 14` size tensor.
### Decoder
The Decoder's job is to **look at the encoded image and generate a caption word by word**.
Since it's generating a sequence, it would need to be a Recurrent Neural Network (RNN). i will use an LSTM.
In a typical setting without Attention, you could simply average the encoded image across all pixels. You could then feed this, with or without a linear transformation, into the Decoder as its first hidden state and generate the caption. Each predicted word is used to generate the next word.
![Decoder without Attention](https://raw.githubusercontent.com/0xprv/PyTorch-Image-to-Caption/refs/heads/main/decoder_no_att.png)
In a setting _with_ Attention, i want the Decoder to be able to **look at different parts of the image at different points in the sequence**. For example, while generating the word `football` in `a man holds a football`, the Decoder would know to focus on – you guessed it – the football!
![Decoding with Attention](https://raw.githubusercontent.com/0xprv/PyTorch-Image-to-Caption/refs/heads/main/decoder_att.png)
Instead of the simple average, i use the _iighted_ average across all pixels, with the iights of the important pixels being greater. This iighted representation of the image can be concatenated with the previously generated word at each step to generate the next word.
### Attention
The Attention network **computes these iights**.
Intuitively, how would you estimate the importance of a certain part of an image? You would need to be aware of the sequence you have generated _so far_, so you can look at the image and decide what needs describing next. For example, after you mention `a man`, it is logical to declare that he is `holding a football`.
This is exactly what the Attention mechanism does – it considers the sequence generated thus far, and _attends_ to the part of the image that needs describing next.
![Attention](https://raw.githubusercontent.com/0xprv/PyTorch-Image-to-Caption/refs/heads/main/att.png)
### Putting it all together
It might be clear by now what our combined network looks like.
![Putting it all together](https://raw.githubusercontent.com/0xprv/PyTorch-Image-to-Caption/refs/heads/main/model.png)
- Once the Encoder generates the encoded image, i transform the encoding to create the initial hidden state `h` (and cell state `C`) for the LSTM Decoder.
- At each decode step,
  - the encoded image and the previous hidden state is used to generate iights for each pixel in the Attention network.
  - the previously generated word and the iighted average of the encoding are fed to the LSTM Decoder to generate the next word.
### Beam Search
i use a linear layer to transform the Decoder's output into a score for each word in the vocabulary.
The straightforward – and greedy – option would be to choose the word with the highest score and use it to predict the next word. But this is not optimal because the rest of the sequence hinges on that first word you choose. If that choice isn't the best, everything that follows is sub-optimal. And it's not just the first word – each word in the sequence has consequences for the ones that succeed it.
It might very ill happen that if you'd chosen the _third_ best word at that first step, and the _second_ best word at the second step, and so on... _that_ would be the best sequence you could generate.
It would be best if i could somehow _not_ decide until i've finished decoding completely, and **choose the sequence that has the highest _overall_ score from a basket of candidate sequences**.
Beam Search does exactly this.
- At the first decode step, consider the top `k` candidates.
- Generate `k` second words for each of these `k` first words.
- Choose the top `k` [first word, second word] combinations considering additive scores.
- For each of these `k` second words, choose `k` third words, choose the top `k` [first word, second word, third word] combinations.
- Repeat at each decode step.
- After `k` sequences terminate, choose the sequence with the best overall score.
![Beam Search example](https://raw.githubusercontent.com/0xprv/PyTorch-Image-to-Caption/refs/heads/main/beam_search.png)
As you can see, some sequences (striked out) may fail early, as they don't make it to the top `k` at the next step. Once `k` sequences (underlined) generate the `<end>` token, i choose the one with the highest score.
# Implementation
The sections below briefly describe the implementation.
They are meant to provide some context, but **details are best understood directly from the code**, which is quite heavily commented.
### Dataset
I'm using the MSCOCO '14 Dataset. You'd need to download the [Training (13GB)](http://images.cocodataset.org/zips/train2014.zip) and [Validation (6GB)](http://images.cocodataset.org/zips/val2014.zip) images.
i will use [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). This zip file contain the captions. You will also find splits and captions for the Flicker8k and Flicker30k datasets, so feel free to use these instead of MSCOCO if the latter is too large for your computer.
### Inputs to model
i will need three inputs.
#### Images
Since i're using a pretrained Encoder, i would need to process the images into the form this pretrained Encoder is accustomed to.
Pretrained ImageNet models available as part of PyTorch's `torchvision` module. [This page](https://pytorch.org/docs/master/torchvision/models.html) details the preprocessing or transformation i need to perform – pixel values must be in the range [0,1] and i must then normalize the image by the mean and standard deviation of the ImageNet images' RGB channels.
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```
Also, PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions.
i will resize all MSCOCO images to 256x256 for uniformity.
Therefore, **images fed to the model must be a `Float` tensor of dimension `N, 3, 256, 256`**, and must be normalized by the aforesaid mean and standard deviation. `N` is the batch size.
#### Captions
Captions are both the target and the inputs of the Decoder as each word is used to generate the next word.
To generate the first word, hoiver, i need a *zeroth* word, `<start>`.
At the last word, i should predict `<end>` the Decoder must learn to predict the end of a caption. This is necessary because i need to know when to stop decoding during inference.
`<start> a man holds a football <end>`
Since i pass the captions around as fixed size Tensors, i need to pad captions (which are naturally of varying length) to the same length with `<pad>` tokens.
`<start> a man holds a football <end> <pad> <pad> <pad>....`
Furthermore, i create a `word_map` which is an index mapping for each word in the corpus, including the `<start>`,`<end>`, and `<pad>` tokens. PyTorch, like other libraries, needs words encoded as indices to look up embeddings for them or to identify their place in the predicted word scores.
`9876 1 5 120 1 5406 9877 9878 9878 9878....`
Therefore, **captions fed to the model must be an `Int` tensor of dimension `N, L`** where `L` is the padded length.
#### Caption Lengths
Since the captions are padded, i would need to keep track of the lengths of each caption. This is the actual length + 2 (for the `<start>` and `<end>` tokens).
Caption lengths are also important because you can build dynamic graphs with PyTorch. i only process a sequence upto its length and don't waste compute on the `<pad>`s.
Therefore, **caption lengths fed to the model must be an `Int` tensor of dimension `N`**.
### Encoder
i use a pretrained ResNet-101 already available in PyTorch's `torchvision` module. Discard the last two layers (pooling and linear layers), since i only need to encode the image, and not classify it.
i do add an `AdaptiveAvgPool2d()` layer to **resize the encoding to a fixed size**. This makes it possible to feed images of variable size to the Encoder. (i did, hoiver, resize our input images to `256, 256` because i had to store them together as a single tensor.)
Since i may want to fine-tune the Encoder, i add a `fine_tune()` method which enables or disables the calculation of gradients for the Encoder's parameters. i **only fine-tune convolutional blocks 2 through 4 in the ResNet**, because the first convolutional block would have usually learned something very fundamental to image processing, such as detecting lines, edges, curves, etc. i don't mess with the foundations.
### Attention
The Attention network is simple – it's composed of only linear layers and a couple of activations.
Separate linear layers **transform both the encoded image (flattened to `N, 14 * 14, 2048`) and the hidden state (output) from the Decoder to the same dimension**, viz. the Attention size. They are then added and ReLU activated. A third linear layer **transforms this result to a dimension of 1**, whereupon i **apply the softmax to generate the iights** `alpha`.
### Decoder
The output of the Encoder is received here and flattened to dimensions `N, 14 * 14, 2048`. This is just convenient and prevents having to reshape the tensor multiple times.
i **initialize the hidden and cell state of the LSTM** using the encoded image with the `init_hidden_state()` method, which uses two separate linear layers.
At the very outset, i **sort the `N` images and captions by decreasing caption lengths**. This is so that i can process only _valid_ timesteps, i.e., not process the `<pad>`s.
i can iterate over each timestep, processing only the colored regions, which are the **_effective_ batch size** `N_t` at that timestep. The sorting allows the top `N_t` at any timestep to align with the outputs from the previous step. At the third timestep, for example, i process only the top 5 images, using the top 5 outputs from the previous step.
This **iteration is performed _manually_ in a `for` loop** with a PyTorch [`LSTMCell`](https://pytorch.org/docs/master/nn.html#torch.nn.LSTM) instead of iterating automatically without a loop with a PyTorch [`LSTM`](https://pytorch.org/docs/master/nn.html#torch.nn.LSTM). This is because i need to execute the Attention mechanism betien each decode step. An `LSTMCell` is a single timestep operation, whereas an `LSTM` would iterate over multiple timesteps continously and provide all outputs at once.
i **compute the iights and attention-iighted encoding** at each timestep with the Attention network. In section `4.2.1` of the paper, they recommend **passing the attention-iighted encoding through a filter or gate**. This gate is a sigmoid activated linear transform of the Decoder's previous hidden state. The authors state that this helps the Attention network put more emphasis on the objects in the image.
i **concatenate this filtered attention-iighted encoding with the embedding of the previous word** (`<start>` to begin), and run the `LSTMCell` to **generate the new hidden state (or output)**. A linear layer **transforms this new hidden state into scores for each word in the vocabulary**, which is stored.
i also store the iights returned by the Attention network at each timestep. You will see why soon enough.
### Some more examples
**The ~~Turing~~ Tommy Test** – you know AI's not really AI because it hasn't watched _The Room_ and doesn't recognize greatness when it sees it.
![](https://raw.githubusercontent.com/0xprv/PyTorch-Image-to-Caption/refs/heads/main/tommy.png)
---
