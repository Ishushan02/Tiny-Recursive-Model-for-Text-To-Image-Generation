# multiModal-arch
Trying out Various Architectures for MultiModal Development.

Adding Modality_Embedding so as to represent that this is Image EMbed and the other is Text Embed


Do QKV for both Image and Text
Apply Rope2-D to Query and Keys
Concatenate Q, K and V from both streams (text and Image) along dim = 2 (sequence Dimension)
After Doing QkV; seperate the attentions blocks for text and image