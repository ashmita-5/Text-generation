# Text generation

## Dataset Selection

**Source:** Hugging Face Datasets Hub <br>
**Link :** https://huggingface.co/datasets/KaungHtetCho/Harry_Potter_LSTM <br>
**Description:** The dataset is correctly hosted on Hugging Face's datasets hub and we have used Harry_Potter datasets which contains a large corpus of text, perfect for language modeling task. The load_dataset function from the datasets library is used to load datasets available in the Hugging Face datasets hub, and the argument is the name of the dataset from the hub.<br>

## Dataset Information

DatasetDict contains three datasets: `train`, `validation`, and `test`. Each dataset consists of text samples, and the number of rows indicates the amount of data in each split. <br>

- **Train Dataset:** <br>
  - Number of rows: 57,435 <br>

- **Validation Dataset:** <br>
  - Number of rows: 5,897 <br>

- **Test Dataset:** <br>
  - Number of rows: 6,589 <br>

## Preprocessing Steps:

### Tokenization:

The text data is tokenized using the basic_english tokenizer from torchtext. Each example in the dataset is tokenized, and the resulting tokens are stored in a new column named 'tokens'.

### Numericalization:

A vocabulary is built from the tokenized dataset, and tokens are replaced with their corresponding indices in the vocabulary. Special tokens like <unk> (unknown) and <eos> (end of sequence) are also added to the vocabulary. The numericalized data is stored as a LongTensor.

## Model Architecture:

The language model is implemented using an LSTM (Long Short-Term Memory) network. The model architecture consists of an embedding layer, an LSTM layer, a dropout layer, and a fully connected layer. The embedding and fully connected layers are responsible for converting token indices to dense representations and predicting the next token, respectively.

### Training Process:

The training process involves iterating through the training dataset, computing the loss, backpropagating the gradients, and updating the model parameters. The model is trained for a specified number of epochs, and the learning rate is adjusted using a learning rate scheduler. The best model is saved based on the validation loss.

## Results
Training perplexity - 79.799
Validation perplexity - 73.351 
Testing perplexity - 88.691

**Output:** Web Application at http://127.0.0.1:5000 <br>

<img width="1690" alt="Screenshot 2024-02-01 at 21 42 59" src="https://github.com/ashmitaphuyal/NLP-Assignment-2/assets/32629216/da30f8fd-d0f6-497a-8dd3-e11e13c8ffda">

<img width="1683" alt="Screenshot 2024-02-01 at 21 43 14" src="https://github.com/ashmitaphuyal/NLP-Assignment-2/assets/32629216/e50c00c8-92c1-4f6e-93cb-1b9a71511b9a">
