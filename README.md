# Text Generation: Stacked LSTM's to generate text in the style Edgar Allen Poe

**Jeremy Chow**

7/15/2019

Goal: Generate text in the style of Edgar Allen Poe, specifically emulating his writing style in the short story dataset from Kaggle "Spooky Author Identification" competition: https://www.kaggle.com/c/spooky-author-identification 

# Model Architecture:
1. Embedding layer
    - Helps model understand 'meaning' of words by mapping them to representative vector space instead of semantic integers
2. Stacked LSTM layers
    - Stacked LSTMs add more depth than additional cells in a single LSTM layer (see paper: https://arxiv.org/abs/1303.5778)
    - The first LSTM layer must have `return sequences` flag set to True in order to pass sequence information to the second LSTM layer instead of just its end states
3. Dense (regression) layer with ReLU activation
4. Dense layer with Softmax activation 
    - Outputs word probability across entire vocab

# Example Output:

**Input:** `First of all I dismembered the corpse`

**Model:**

"First of all i dismembered the corpse which is profound since in the first place he appeared at first so suddenly as any matter no answer was impossible to find my sake he now happened to be sure it was he suspected or caution or gods and some voice held forth upon"