//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

/*
 * The size of the hash table for the vocabulary.
 * The vocabulary won't be allowed to grow beyond 70% of this number.
 * For instance, if the hash table has 30M entries, then the maximum
 * vocab size is 21M. This is to minimize the occurrence (and performance
 * impact) of hash collisions.
 */
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

/**
 * ======== vocab_word ========
 * Properties:
 *   cn - The word frequency (number of times it appears).
 *   word - The actual string word.
 */
struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

/*
 * ======== Global Variables ========
 *
 */
char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

/*
 * ======== vocab ========
 * This array will hold all of the words in the vocabulary.
 * This is internal state.
 */
struct vocab_word *vocab;

int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

/*
 * ======== vocab_hash ========
 * This array is the hash table for the vocabulary. Word strings are hashed
 * to a hash code (an integer), then the hash code is used as the index into
 * 'vocab_hash', to retrieve the index of the word within the 'vocab' array.
 */
int *vocab_hash;

/*
 * ======== vocab_max_size ========
 * This is not a limit on the number of words in the vocabulary, but rather
 * a chunk size for allocating the vocabulary table. The vocabulary table will
 * be expanded as necessary, and is allocated, e.g., 1,000 words at a time.
 *
 * ======== vocab_size ========
 * Stores the number of unique words in the vocabulary. 
 * This is not a parameter, but rather internal state. 
 *
 * ======== layer1_size ========
 * This is the number of features in the word vectors.
 * It is the number of neurons in the hidden layer of the model.
 */
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;

/*
 *
 */
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;

/*
 * ======== alpha ========
 * TODO - This is a learning rate parameter.
 *
 * ======== starting_alpha ========
 *
 * ======== sample ========
 * This parameter controls the subsampling of frequent words.
 * Smaller values of 'sample' mean words are less likely to be kept.
 * Set 'sample' to 0 to disable subsampling.
 * See the comments in the subsampling section for more details.
 */
real alpha = 0.025, starting_alpha, sample = 1e-3;

/*
 * IMPORTANT - Note that the weight matrices are stored as 1D arrays, not
 * 2D matrices, so to access row 'i' of syn0, the index is (i * layer1_size).
 * 
 * ======== syn0 ========
 * This is the hidden layer weights (which is also the word vectors!)
 *
 * ======== syn1 ========
 * This is the output layer weights *if using heirarchical softmax*
 *
 * ======== syn1neg ========
 * This is the output layer weights *if using negative sampling*
 *
 * ======== expTable ========
 * Stores precalcultaed activations for the output layer.
 */
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

/**
 * ======== InitUnigramTable ========
 * This table is used to implement negative sampling.
 * Each word is given a weight equal to it's frequency (word count) raised to
 * the 3/4 power. The probability for a selecting a word is just its weight 
 * divided by the sum of weights for all words. 
 *
 * Note that the vocabulary has been sorted by word count, descending, such 
 * that we will go through the vocabulary from most frequent to least.
 */
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  
  double d1, power = 0.75;
  
  // Allocate the table. It's bigger than the vocabulary, because words will
  // appear in it multiple times based on their frequency.
  // Every vocab word appears at least once in the table.
  // The size of the table relative to the size of the vocab dictates the 
  // resolution of the sampling. A larger unigram table means the negative 
  // samples will be selected with a probability that more closely matches the
  // probability calculated by the equation.
  table = (int *)malloc(table_size * sizeof(int));
  
  // Calculate the denominator, which is the sum of weights for all words.
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  
  // 'i' is the vocabulary index of the current word, whereas 'a' will be
  // the index into the unigram table.
  i = 0;
  
  // Calculate the probability that we choose word 'i'. This is a fraction
  // betwee 0 and 1.
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  
  // Loop over all positions in the table.
  for (a = 0; a < table_size; a++) {
    
    // Store word 'i' in this position. Word 'i' will appear multiple times
    // in the table, based on its frequency in the training data.    
    table[a] = i;
    
    // If the fraction of the table we have filled is greater than the
    // probability of choosing this word, then move to the next word.
    if (a / (double)table_size > d1) {
      // Move to the next word.
      i++;
      
      // Calculate the probability for the new word, and accumulate it with 
      // the probabilities of all previous words, so that we can compare d1 to
      // the percentage of the table that we have filled.
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    // Don't go past the end of the vocab. 
    // The total weights for all words should sum up to 1, so there shouldn't
    // be any extra space at the end of the table. Maybe it's possible to be
    // off by 1, though? Or maybe this is just precautionary.
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

/**
 * ======== ReadWord ========
 * Reads a single word from a file, assuming space + tab + EOL to be word 
 * boundaries.
 *
 * Parameters:
 *   word - A char array allocated to hold the maximum length string.
 *   fin  - The training file.
 */
void ReadWord(char *word, FILE *fin) {
  
  // 'a' will be the index into 'word'.
  int a = 0, ch;
  
  // Read until the end of the word or the end of the file.
  while (!feof(fin)) {
  
    // Get the next character.
    ch = fgetc(fin);
    
    // ASCII Character 13 is a carriage return 'CR' whereas character 10 is 
    // newline or line feed 'LF'.
    if (ch == 13) continue;
    
    // Check for word boundaries...
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      // If the word has at least one character, we're done.
      if (a > 0) {
        // Put the newline back before returning so that we find it next time.
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      // If the word is empty and the character is newline, treat this as the
      // end of a "sentence" and mark it with the token </s>.
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      // If the word is empty and the character is tab or space, just continue
      // on to the next character.     
      } else continue;
    }
    
    // If the character wasn't space, tab, CR, or newline, add it to the word.
    word[a] = ch;
    a++;
    
    // If the word's too long, truncate it, but keep going till we find the end
    // of it.
    if (a >= MAX_STRING - 1) a--;   
  }
  
  // Terminate the string with null.
  word[a] = 0;
}

/**
 * ======== GetWordHash ========
 * Returns hash value of a word. The hash is an integer between 0 and 
 * vocab_hash_size (default is 30E6).
 *
 * For example, the word 'hat':
 * hash = ((((h * 257) + a) * 257) + t) % 30E6
 */
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

/**
 * ======== SearchVocab ========
 * Lookup the index in the 'vocab' table of the given 'word'.
 * Returns -1 if the word is not found.
 * This function uses a hash table for fast lookup.
 */
int SearchVocab(char *word) {
  // Compute the hash value for 'word'.
  unsigned int hash = GetWordHash(word);
  
  // Lookup the index in the hash table, handling collisions as needed.
  // See 'AddWordToVocab' to see how collisions are handled.
  while (1) {
    // If the word isn't in the hash table, it's not in the vocab.
    if (vocab_hash[hash] == -1) return -1;
    
    // If the input word matches the word stored at the index, we're good,
    // return the index.
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    
    // Otherwise, we need to scan through the hash table until we find it.
    hash = (hash + 1) % vocab_hash_size;
  }
  
  // This will never be reached.
  return -1;
}

/**
 * ======== ReadWordIndex ========
 * Reads the next word from the training file, and returns its index into the
 * 'vocab' table.
 */
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

/**
 * ======== AddWordToVocab ========
 * Adds a new word to the vocabulary (one that hasn't been seen yet).
 */
int AddWordToVocab(char *word) {
  // Measure word length.
  unsigned int hash, length = strlen(word) + 1;
  
  // Limit string length (default limit is 100 characters).
  if (length > MAX_STRING) length = MAX_STRING;
  
  // Allocate and store the word string.
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  
  // Initialize the word frequency to 0.
  vocab[vocab_size].cn = 0;
  
  // Increment the vocabulary size.
  vocab_size++;
  
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  
  // Add the word to the 'vocab_hash' table so that we can map quickly from the
  // string to its vocab_word structure. 
  
  // Hash the word to an integer between 0 and 30E6.
  hash = GetWordHash(word);
  
  // If the spot is already taken in the hash table, find the next empty spot.
  while (vocab_hash[hash] != -1) 
    hash = (hash + 1) % vocab_hash_size;
  
  // Map the hash code to the index of the word in the 'vocab' array.  
  vocab_hash[hash] = vocab_size - 1;
  
  // Return the index of the word in the 'vocab' array.
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

/**
 * ======== SortVocab ========
 * Sorts the vocabulary by frequency using word counts, and removes words that
 * occur fewer than 'min_count' times in the training text.
 * 
 * Removing words from the vocabulary requires recomputing the hash table.
 */
void SortVocab() {
  int a, size;
  unsigned int hash;
  
  /*
   * Sort the vocabulary by number of occurrences, in descending order. 
   *
   * Keep </s> at the first position by sorting starting from index 1.
   *
   * Sorting the vocabulary this way causes the words with the fewest 
   * occurrences to be at the end of the vocabulary table. This will allow us
   * to free the memory associated with the words that get filtered out.
   */
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  
  // Clear the vocabulary hash table.
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  
  // Store the initial vocab size to use in the for loop condition.
  size = vocab_size;
  
  // Recompute the number of training words.
  train_words = 0;
  
  // For every word currently in the vocab...
  for (a = 0; a < size; a++) {
    // If it occurs fewer than 'min_count' times, remove it from the vocabulary.
    if ((vocab[a].cn < min_count) && (a != 0)) {
      // Decrease the size of the new vocabulary.
      vocab_size--;
      
      // Free the memory associated with the word string.
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
   
  // Reallocate the vocab array, chopping off all of the low-frequency words at
  // the end of the table.
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

/**
 * ======== CreateBinaryTree ========
 * Create binary Huffman tree using the word counts.
 * Frequent words will have short unique binary codes.
 * Huffman encoding is used for lossless compression.
 * For each vocabulary word, the vocab_word structure includes a `point` array, 
 * which is the list of internal tree nodes which:
 *   1. Define the path from the root to the leaf node for the word.
 *   2. Each correspond to a row of the output matrix.
 * The `code` array is a list of 0s and 1s which specifies whether each output
 * in `point` should be trained to output 0 or 1.
 */
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH]; // Default is 40
  
  // Note that calloc initializes these arrays to 0.
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  
  // The count array is twice the size of the vocabulary, plus one.
  //   - The first half of `count` becomes a list of the word counts 
  //     for each word in the vocabulary. We do not modify this part of the
  //     list.
  //   - The second half of `count` is set to a large positive integer (1 
  //     quadrillion). When we combine two trees under a word (e.g., word_id 
  //     13), then we place the total weight of those subtrees into the word's
  //     position in the second half (e.g., count[vocab_size + 13]).
  //     
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  
  // `pos1` and `pos2` are indeces into the `count` array.
  //   - `pos1` starts at the middle of `count` (the end of the list of word
  //     counts) and moves left.
  //   - `pos2` starts at the beginning of the list of large integers and moves
  //     right.
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  
  
  /* ===============================
   *   Step 1: Create Huffman Tree
   * ===============================
   * [Original Comment] Following algorithm constructs the Huffman tree by 
   * adding one node at a time
   * 
   * The Huffman coding algorithm starts with every node as its own tree, and
   * then combines the two smallest trees on each step. The weight of a tree is
   * the sum of the word counts for the words it contains. 
   * 
   * Once the tree is constructed, you can use the `parent_node` array to 
   * navigate it. For the word at index 13, for example, you would look at 
   * parent_node[13], and then parent_node[parent_node[13]], and so on, till
   * you reach the root.
   *
   * A Huffman tree stores all of the words in the vocabulary at the leaves.
   * Frequent words have short paths, and infrequent words have long paths.
   * Here, we are also associating each internal node of the tree with a 
   * row of the output matrix. Every time we combine two trees and create a 
   * new node, we give it a row in the output matrix.
   */  
  
  // The number of tree combinations needed is equal to the size of the vocab,
  // minus 1.
  for (a = 0; a < vocab_size - 1; a++) {

    // First, find two smallest nodes 'min1, min2'
    // Find min1 (at index `min1i`)
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    
    // Find min2 (at index `min2i`).
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    
    // Calculate the combined weight. We could be combining two words, a word
    // and a tree, or two trees.
    count[vocab_size + a] = count[min1i] + count[min2i];
    
    // Store the path for working back up the tree. 
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    
    // binary[min1i] = 0; // This is implied.
    // min1 is the (left?) node and is labeled '0', min2 is the (right?) node
    // and is labeled '1'.
    binary[min2i] = 1;
  }

  /* ==========================================
   *    Step 2: Define Samples for Each Word
   * ==========================================
   * [Original Comment] Now assign binary code to each vocabulary word
   * 
   *  vocab[word]
   *    .code - A variable-length string of 0s and 1s.
   *    .point - A variable-length array of output row indeces.
   *    .codelen - The length of the `code` array. 
   *               The point array has length `codelen + 1`.
   * 
   */  
    
  // For each word in the vocabulary...
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0; // `i` stores the code length.
    
    // Construct the binary code...
    //   `code` stores 1s and 0s.
    //   `point` stores indeces.
    // This loop works backwards from the leaf, so the `code` and `point` 
    // lists end up in reverse order.
    while (1) {
      // Lookup whether this is on the left or right of its parent node.
      code[i] = binary[b];
      
      // Note: point[0] always holds the word iteself...
      point[i] = b;
      
      // Increment the code length.
      i++;
      
      // This will always return an index in the second half of the array.
      b = parent_node[b];
      
      // We've reached the root when...
      if (b == vocab_size * 2 - 2) break;
    }
    
    // Record the code length (the length of the `point` list).
    vocab[a].codelen = i;
    
    // The root node is at row `vocab_size - 2` of the output matrix. 
    vocab[a].point[0] = vocab_size - 2;
    
    // For each bit in this word's code...
    for (b = 0; b < i; b++) {
      // Reverse the code in `code` and store it in `vocab[a].code`
      vocab[a].code[i - b - 1] = code[b];
      
      // Store the row indeces of the internal nodes leading to this word.
      // These are the set of outputs which will be trained every time
      // this word is encountered in the training data as an output word.
      vocab[a].point[i - b] = point[b] - vocab_size;      
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

/**
 * ======== LearnVocabFromTrainFile ========
 * Builds a vocabulary from the words found in the training file.
 *
 * This function will also build a hash table which allows for fast lookup
 * from the word string to the corresponding vocab_word object.
 *
 * Words that occur fewer than 'min_count' times will be filtered out of
 * vocabulary.
 */
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  
  // Populate the vocab table with -1s.
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  
  // Open the training file.
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  
  vocab_size = 0;
  
  // The special token </s> is used to mark the end of a sentence. In training,
  // the context window does not go beyond the ends of a sentence.
  // 
  // Add </s> explicitly here so that it occurs at position 0 in the vocab. 
  AddWordToVocab((char *)"</s>");
  
  while (1) {
    // Read the next word from the file into the string 'word'.
    ReadWord(word, fin);
    
    // Stop when we've reached the end of the file.
    if (feof(fin)) break;
    
    // Count the total number of tokens in the training text.
    train_words++;
    
    // Print progress at every 100,000 words.
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    
    // Look up this word in the vocab to see if we've already added it.
    i = SearchVocab(word);
    
    // If it's not in the vocab...
    if (i == -1) {
      // ...add it.
      a = AddWordToVocab(word);
      
      // Initialize the word frequency to 1.
      vocab[a].cn = 1;
    
    // If it's already in the vocab, just increment the word count.
    } else vocab[i].cn++;
    
    // If the vocabulary has grown too large, trim out the most infrequent 
    // words. The vocabulary is considered "too large" when it's filled more
    // than 70% of the hash table (this is to try and keep hash collisions
    // down).
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  
  // Sort the vocabulary in descending order by number of word occurrences.
  // Remove (and free the associated memory) for all the words that occur
  // fewer than 'min_count' times.
  SortVocab();
  
  // Report the final vocabulary size, and the total number of words 
  // (excluding those filtered from the vocabulary) in the training set.
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

/**
 * ======== InitNet ========
 *
 */
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  
  // Allocate the hidden layer of the network, which is what becomes the word vectors.
  // The variable for this layer is 'syn0'.
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  
  // If we're using hierarchical softmax for training...
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  
  // If we're using negative sampling for training...
  if (negative>0) {
    // Allocate the output layer of the network. 
    // The variable for this layer is 'syn1neg'.
    // This layer has the same size as the hidden layer, but is the transpose.
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    // Set all of the weights in the output layer to 0.
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  
  // Randomly initialize the weights for the hidden layer (word vector layer).
  // TODO - What's the equation here?
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  
  // Create a binary tree for Huffman coding.
  // TODO - As best I can tell, this is only used for hierarchical softmax training...
  CreateBinaryTree();
}

/**
 * ======== TrainModelThread ========
 * This function performs the training of the model.
 */
void *TrainModelThread(void *id) {

  /*
   * word - Stores the index of a word in the vocab table.
   * word_count - Stores the total number of training words processed.
   */
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  
  // neu1 is only used by the CBOW architecture.
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  
  // neu1e is used by both architectures.
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  
  
  // Open the training file and seek to the portion of the file that this 
  // thread is responsible for.
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  
  // This loop covers the whole training operation...
  while (1) {
    
    /*
     * ======== Variables ========
     *       iter - This is the number of training epochs to run; default is 5.
     * word_count - The number of input words processed.
     * train_words - The total number of words in the training text (not 
     *               including words removed from the vocabuly by ReduceVocab).
     */
    
    // This block prints a progress update, and also adjusts the training 
    // 'alpha' parameter.
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      
      last_word_count = word_count;
      
      // The percentage complete is based on the total number of passes we are
      // doing and not just the current pass.      
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         // Percent complete = [# of input words processed] / 
         //                      ([# of passes] * [# of words in a pass])
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      
      // Update alpha to: [initial alpha] * [percent of training remaining]
      // This means that alpha will gradually decrease as we progress through 
      // the training text.
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      // Don't let alpha go below [initial alpha] * 0.0001.
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    
    // This 'if' block retrieves the next sentence from the training text and
    // stores it in 'sen'.
    //
    // Note the main while loop do two things:
    // 1. run this 'if' block to prepare sentence, and
    // 2. move one word forward in current sentence, i.e., 'sentence_position++',
    //    prepare context and target, and do the forward computation and backward
    //    propagation
    //
    // the if statement here check whether the training process just start or just
    // finished the last sentence, and need to prepare a new sentence to do part 2 
    if (sentence_length == 0) {
      while (1) {
        // Read the next word from the training data and lookup its index in 
        // the vocab table. 'word' is the word's vocab index.
        word = ReadWordIndex(fi);
        
        if (feof(fi)) break;
        
        // If the word doesn't exist in the vocabulary, skip it.
        if (word == -1) continue;
        
        // Track the total number of training words processed.
        word_count++;
        
        // 'vocab' word 0 is a special token "</s>" which indicates the end of 
        // a sentence.
        if (word == 0) break;
        
        /* 
         * =================================
         *   Subsampling of Frequent Words
         * =================================
         * This code randomly discards training words, but is designed to 
         * keep the relative frequencies the same. That is, less frequent
         * words will be discarded less often. 
         *
         * We first calculate the probability that we want to *keep* the word;
         * this is the value 'ran'. Then, to decide whether to keep the word,
         * we generate a random fraction (0.0 - 1.0), and if 'ran' is smaller
         * than this number, we discard the word. This means that the smaller 
         * 'ran' is, the more likely it is that we'll discard this word. 
         *
         * The quantity (vocab[word].cn / train_words) is the fraction of all 
         * the training words which are 'word'. Let's represent this fraction
         * by x.
         *
         * Using the default 'sample' value of 0.001, the equation for ran is:
         *   ran = (sqrt(x / 0.001) + 1) * (0.001 / x)
         * 
         * You can plot this function to see it's behavior; it has a curved 
         * L shape.
         * 
         * Here are some interesting points in this function (again this is
         * using the default sample value of 0.001).
         *   - ran = 1 (100% chance of being kept) when x <= 0.0026.
         *      - That is, any word which is 0.0026 of the words *or fewer* 
         *        will be kept 100% of the time. Only words which represent 
         *        more than 0.26% of the total words will be subsampled.
         *   - ran = 0.5 (50% chance of being kept) when x = 0.00746. 
         *   - ran = 0.033 (3.3% chance of being kept) when x = 1.
         *       - That is, if a word represented 100% of the training set
         *         (which of course would never happen), it would only be
         *         kept 3.3% of the time.
         *
         * NOTE: Seems like it would be more efficient to pre-calculate this 
         *       probability for each word and store it in the vocab table...
         *
         * Words that are discarded by subsampling aren't added to our training
         * 'sentence'. This means the discarded word is neither used as an 
         * input word or a context word for other inputs.
         */
        if (sample > 0) {
          // Calculate the probability of keeping 'word'.
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          
          // Generate a random number.
          // The multiplier is 25.xxx billion, so 'next_random' is a 64-bit integer.
          next_random = next_random * (unsigned long long)25214903917 + 11;

          // If the probability is less than a random fraction, discard the word.
          //
          // (next_random & 0xFFFF) extracts just the lower 16 bits of the 
          // random number. Dividing this by 65536 (2^16) gives us a fraction
          // between 0 and 1. So the code is just generating a random fraction.
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        
        // If we kept the word, add it to the sentence.
        sen[sentence_length] = word;
        sentence_length++;
        
        // Verify the sentence isn't too long.
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    
    // Get the next word in the sentence. The word is represented by its index
    // into the vocab table.
    word = sen[sentence_position];
    
    if (word == -1) continue;
    
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    
    // This is a standard random integer generator, as seen here:
    // https://en.wikipedia.org/wiki/Linear_congruential_generator
    next_random = next_random * (unsigned long long)25214903917 + 11;
    
    // 'b' becomes a random integer between 0 and 'window' - 1.
    // This is the amount we will shrink the window size by.
    b = next_random % window;
    
    /* 
     * ====================================
     *        CBOW Architecture
     * ====================================
     * sen - This is the array of words in the sentence. Subsampling has 
     *       already been applied. Words are represented by their ids.
     *
     * sentence_position - This is the index of the current input word.
     *
     * a - Offset into the current window, relative to the window start.
     *     a will range from 0 to (window * 2) 
     *
     * b - The amount to shrink the context window by.
     *
     * c - 'c' is a scratch variable used in two unrelated ways:
     *       1. It's first used as the index of the current context word 
     *          within the sentence (the `sen` array).
     *       2. It's then used as the for-loop variable for calculating
     *          vector dot-products and other arithmetic.
     *
     * syn0 - The hidden layer weights. Note that the weights are stored as a
     *        1D array, so word 'i' is found at (i * layer1_size).
     *
     * target - The output word we're working on. If it's the positive sample
     *          then `label` is 1. `label` is 0 for negative samples.  
     *          Note: `target` and `label` are only used in negative sampling,
     *                and not HS.
     *
     * neu1 - This vector will hold the *average* of all of the context word
     *        vectors. This is the output of the hidden layer for CBOW.
     *
     * neu1e - Holds the gradient for updating the hidden layer weights.
     *         It's a vector of length 300, not a matrix.
     *         This same gradient update is applied to all context word 
     *         vectors.
     */
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      
      // This loop will sum together the word vectors for all of the context
      // words.
      //
      // Loop over the positions in the context window (skipping the word at
      // the center). 'a' is just the offset within the window, it's not the 
      // index relative to the beginning of the sentence.
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {

        // Convert the window offset 'a' into an index 'c' into the sentence 
        // array.
        c = sentence_position - window + a;
        
        // Verify c isn't outisde the bounds of the sentence.
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        
        // Get the context word. That is, get the id of the word (its index in
        // the vocab table).
        last_word = sen[c];
        
        // At this point we have two words identified:
        //   'word' - The word (word ID) at our current position in the 
        //            sentence (in the center of a context window).
        //   'last_word' - The word (word ID) at a position within the context
        //                 window.       
        
        // Verify that the word exists in the vocab
        if (last_word == -1) continue;
        
        // Add the word vector for this context word to the running sum in 
        // neur1.
        // `layer1_size` is 300, `neu1` is length 300
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        
        // Count the number of context words.
        cw++;
      }
      
      // Skip if there were somehow no context words.
      if (cw) {
        
        // neu1 was the sum of the context word vectors, and now becomes
        // their average. 
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        
        // // HIERARCHICAL SOFTMAX
        // vocab[word]
        //   .point - A variable-length list of row ids, which are the output
        //            rows to train on.
        //   .code - A variable-length list of 0s and 1s, which are the desired
        //           labels for the outputs in `point`.
        //   .codelen - The length of the `code` array for this 
        //              word.
        // 
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          // point[d] is the index of a row of the ouput matrix.
          // l2 is the index of that word in the output layer weights (syn1).
          l2 = vocab[word].point[d] * layer1_size;
          
          // Propagate hidden -> output
          // neu1 is the average of the context words from the hidden layer.
          // This loop computes the dot product between neu1 and the output
          // weights for the output word at point[d].
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          
          // Apply the sigmoid activation to the current output neuron.
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          
          // 'g' is the error multiplied by the learning rate.
          // The error is (label - f), so label = (1 - code), meaning if
          // code is 0, then this is a positive sample and vice versa.
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        
        // NEGATIVE SAMPLING
        // Rather than performing backpropagation for every word in our 
        // vocabulary, we only perform it for the positive sample and a few
        // negative samples (the number of words is given by 'negative').
        // These negative words are selected using a "unigram" distribution, 
        // which is generated in the function InitUnigramTable.        
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          // On the first iteration, we're going to train the positive sample.
          if (d == 0) {
            target = word;
            label = 1;
          
          // On the other iterations, we'll train the negative samples.
          } else {
            // Pick a random word to use as a 'negative sample'; do this using 
            // the unigram table.
            
            // Get a random integer.              
            next_random = next_random * (unsigned long long)25214903917 + 11;
            
            // 'target' becomes the index of the word in the vocab to use as
            // the negative sample.            
            target = table[(next_random >> 16) % table_size];
            
            // If the target is the special end of sentence token, then just
            // pick a random word from the vocabulary instead.            
            if (target == 0) target = next_random % (vocab_size - 1) + 1;

            // Don't use the positive sample as a negative sample!            
            if (target == word) continue;
            
            // Mark this as a negative example.
            label = 0;
          }
          
          // At this point, target might either be the positive sample or a 
          // negative sample, depending on the value of `label`.
          
          // Get the index of the target word in the output layer.
          l2 = target * layer1_size;
          
          // Calculate the dot product between:
          //   neu1 - The average of the context word vectors.
          //   syn1neg[l2] - The output weights for the target word.
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];

          // This block does two things:
          //   1. Calculates the output of the network for this training
          //      pair, using the expTable to evaluate the output layer
          //      activation function.
          //   2. Calculate the error at the output, stored in 'g', by
          //      subtracting the network output from the desired output, 
          //      and finally multiply this by the learning rate.          
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          
          // Multiply the error by the output layer weights.
          // (I think this is the gradient calculation?)
          // Accumulate these gradients over all of the negative samples.          
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];

          // Update the output layer weights by multiplying the output error
          // by the average of the context word vectors.
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
         
        // hidden -> in
        // Backpropagate the error to the hidden layer (the word vectors).
        // This code is used both for heirarchical softmax and for negative
        // sampling.
        //
        // Loop over the positions in the context window (skipping the word at
        // the center). 'a' is just the offset within the window, it's not 
        // the index relative to the beginning of the sentence.
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          // Convert the window offset 'a' into an index 'c' into the sentence 
          // array.
          c = sentence_position - window + a;
          
          // Verify c isn't outisde the bounds of the sentence.
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          
          // Get the context word. That is, get the id of the word (its index in
          // the vocab table).
          last_word = sen[c];
          
          // Verify word exists in vocab.
          if (last_word == -1) continue;
          
          // Note that `c` is no longer the sentence position, it's just a 
          // for-loop index.
          // Add the gradient in the vector `neu1e` to the word vector for
          // the current context word.
          // syn0[last_word * layer1_size] <-- Accesses the word vector.
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } 
    /* 
     * ====================================
     *        Skip-gram Architecture
     * ====================================
     * sen - This is the array of words in the sentence. Subsampling has 
     *       already been applied. Words are represented by their ids.
     *
     * sentence_position - This is the index of the current input word.
     *
     * a - Offset into the current window, relative to the window start.
     *     a will range from 0 to (window * 2) 
     *
     * b - The amount to shrink the context window by.
     *
     * c - 'c' is a scratch variable used in two unrelated ways:
     *       1. It's first used as the index of the current context word 
     *          within the sentence (the `sen` array).
     *       2. It's then used as the for-loop variable for calculating
     *          vector dot-products and other arithmetic.
     *
     * syn0 - The hidden layer weights. Note that the weights are stored as a
     *        1D array, so word 'i' is found at (i * layer1_size).
     *
     * l1 - Index into the hidden layer (syn0). Index of the start of the
     *      weights for the current input word.
     *
     * target - The output word we're working on. If it's the positive sample
     *          then `label` is 1. `label` is 0 for negative samples.
     *          Note: `target` and `label` are only used in negative sampling,
     *                and not HS.     
     */
    else {  
      // Loop over the positions in the context window (skipping the word at
      // the center). 'a' is just the offset within the window, it's not 
      // the index relative to the beginning of the sentence.
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        
        // Convert the window offset 'a' into an index 'c' into the sentence 
        // array.
        c = sentence_position - window + a;
        
        // Verify c isn't outisde the bounds of the sentence.
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        
        // Get the context word. That is, get the id of the word (its index in
        // the vocab table).
        last_word = sen[c];
        
        // At this point we have two words identified:
        //   'word' - The word at our current position in the sentence (in the
        //            center of a context window).
        //   'last_word' - The word at a position within the context window.
        
        // Verify that the word exists in the vocab (I don't think this should
        // ever be the case?)
        if (last_word == -1) continue;
        
        // Calculate the index of the start of the weights for 'last_word'.
        l1 = last_word * layer1_size;
        
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        
        // NEGATIVE SAMPLING
        // Rather than performing backpropagation for every word in our 
        // vocabulary, we only perform it for a few words (the number of words 
        // is given by 'negative').
        // These words are selected using a "unigram" distribution, which is generated
        // in the function InitUnigramTable
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          // On the first iteration, we're going to train the positive sample.
          if (d == 0) {
            target = word;
            label = 1;
          // On the other iterations, we'll train the negative samples.
          } else {
            // Pick a random word to use as a 'negative sample'; do this using 
            // the unigram table.
            
            // Get a random integer.
            next_random = next_random * (unsigned long long)25214903917 + 11;
            
            // 'target' becomes the index of the word in the vocab to use as
            // the negative sample.
            target = table[(next_random >> 16) % table_size];
            
            // If the target is the special end of sentence token, then just
            // pick a random word from the vocabulary instead.
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            
            // Don't use the positive sample as a negative sample!
            if (target == word) continue;
            
            // Mark this as a negative example.
            label = 0;
          }
          
          // Get the index of the target word in the output layer.
          l2 = target * layer1_size;
          
          // At this point, our two words are represented by their index into
          // the layer weights.
          // l1 - The index of our input word within the hidden layer weights.
          // l2 - The index of our output word within the output layer weights.
          // label - Whether this is a positive (1) or negative (0) example.
          
          // Calculate the dot-product between the input words weights (in 
          // syn0) and the output word's weights (in syn1neg).
          // Note that this calculates the dot-product manually using a for
          // loop over the vector elements!
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          
          // This block does two things:
          //   1. Calculates the output of the network for this training
          //      pair, using the expTable to evaluate the output layer
          //      activation function.
          //   2. Calculate the error at the output, stored in 'g', by
          //      subtracting the network output from the desired output, 
          //      and finally multiply this by the learning rate.
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          
          // Multiply the error by the output layer weights.
          // Accumulate these gradients over the negative samples and the one
          // positive sample.
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          
          // Update the output layer weights by multiplying the output error
          // by the hidden layer weights.
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Once the hidden layer gradients for the negative samples plus the 
        // one positive sample have been accumulated, update the hidden layer
        // weights. 
        // Note that we do not average the gradient before applying it.
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    
    // Advance to the next word in the sentence.
    sentence_position++;
    
    // Check if we've reached the end of the sentence.
    // If so, set sentence_length to 0 and we'll read a new sentence at the
    // beginning of this loop.
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

/**
 * ======== TrainModel ========
 * Main entry point to the training process.
 */
void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  
  printf("Starting training using file %s\n", train_file);
  
  starting_alpha = alpha;
  
  // Either load a pre-existing vocabulary, or learn the vocabulary from 
  // the training file.
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  
  // Save the vocabulary.
  if (save_vocab_file[0] != 0) SaveVocab();
  
  // Stop here if no output_file was specified.
  if (output_file[0] == 0) return;
  
  // Allocate the weight matrices and initialize them.
  InitNet();

  // If we're using negative sampling, initialize the unigram table, which
  // is used to pick words to use as "negative samples" (with more frequent
  // words being picked more often).  
  if (negative > 0) InitUnigramTable();
  
  // Record the start time of training.
  start = clock();
  
  // Run training, which occurs in the 'TrainModelThread' function.
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  
  
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  
  // Allocate the vocabulary table.
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  
  // Allocate the hash table for mapping word strings to word entries.
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
   
  /*
   * ======== Precomputed Exp Table ========
   * To calculate the softmax output, they use a table of values which are
   * pre-computed here.
   *
   * From the top of this file:
   *   #define EXP_TABLE_SIZE 1000
   *   #define MAX_EXP 6
   *
   * First, let's look at this inner term:
   *     i / (real)EXP_TABLE_SIZE * 2 - 1
   * This is just a straight line that goes from -1 to +1.
   *    (0, -1.0), (1, -0.998), (2, -0.996), ... (999, 0.998), (1000, 1.0).
   *
   * Next, multiplying this by MAX_EXP = 6, it causes the output to range
   * from -6 to +6 instead of -1 to +1.
   *    (0, -6.0), (1, -5.988), (2, -5.976), ... (999, 5.988), (1000, 6.0).
   *
   * So the total input range of the table is 
   *    Range = MAX_EXP * 2 = 12
   * And the increment on the inputs is
   *    Increment = Range / EXP_TABLE_SIZE = 0.012
   *
   * Let's say we want to compute the output for the value x = 0.25. How do
   * we calculate the position in the table?
   *    index = (x - -MAX_EXP) / increment
   * Which we can re-write as:
   *    index = (x + MAX_EXP) / (range / EXP_TABLE_SIZE)
   *          = (x + MAX_EXP) / ((2 * MAX_EXP) / EXP_TABLE_SIZE)
   *          = (x + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)
   *
   * The last form is what we find in the code elsewhere for using the table:
   *    expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
   * 
   */

  // Allocate the table, 1000 floats.
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  
  // For each position in the table...
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    
    // Calculate the output of e^x for values in the range -6.0 to +6.0.
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    
    // Currently the table contains the function exp(x).
    // We are going to replace this with exp(x) / (exp(x) + 1), which is
    // just the sigmoid activation function! 
    // Note that 
    //    exp(x) / (exp(x) + 1) 
    // is equivalent to 
    //    1 / (1 + exp(-x))
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  
  TrainModel();
  return 0;
}
