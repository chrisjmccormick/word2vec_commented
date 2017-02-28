
#word2vec_commented
This project is a functionally unaltered version of Google's published word2vec implementation, but which includes source comments.

This is a work in progress.

## word2vec Model Training

word2vec training occurs in word2vec.c

* main() - Entry point to the script.
    * Parses the command line arguments.
* TrainModel() - Main entry point to the training process.


### Building the Vocabulary
`word2vec.c` includes code for constructing a vocabulary from an input text file.

The code supports fast lookup of vocab words through a hash table, which maps word strings to their respective `vocab_word` object. 

The completed vocabulary consists of the following:

* `vocab_word` - A structure containing a word and its metadata, such as its frequency (word count) in the training text.
* `vocab` - The array of `vocab_word` objects for every word.
* `vocab_hash` - A hash table which maps word hash codes to the index of the word in the `vocab` array. The word hash is calculated using the `GetWordHash` function.

Learning the vocabulary starts with the `LearnVocabFromTrainFile` function.