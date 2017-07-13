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

real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

/**
 * ======== InitUnigramTable ========
 */
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    // [Chris] - The table may contain multiple elements which hold value 'i'.
    //           
    table[a] = i;
    // [Chris] - If the fraction of the table we have filled is greater than the
    //           fraction of this words weight / all word weights, then move to the next word.
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
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
 * The vocab_word structure contains a field for the 'code' for the word.
 */
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
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
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
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
    // This block prints a progress update, and also adjusts the training 
    // 'alpha' parameter.
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    
    // This 'if' block retrieves the next sentence from the training text and
    // stores it in 'sen'.
    // TODO - Under what condition would sentence_length not be zero?
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
    
    next_random = next_random * (unsigned long long)25214903917 + 11;
    
    // 'b' becomes a random integer between 0 and 'window'.
    // This is the amount we will shrink the window size by.
    b = next_random % window;
    
    /* 
     * ====================================
     *        CBOW Architecture
     * ====================================
     */
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } 
    /* 
     * ====================================
     *        Skip-gram Architecture
     * ====================================
     * sen - This is the array of words in the sentence. Subsampling has already been
     *       applied. I don't know what the word representation is...
     *
     * sentence_position - This is the index of the current input word.
     *
     * a - Offset into the current window, relative to the window start.
     *     a will range from 0 to (window * 2) (TODO - not sure if it's inclusive or
     *      not).
     *
     * c - 'c' is the index of the current context word *within the sentence*
     *
     * syn0 - The hidden layer weights.
     *
     * l1 - Index into the hidden layer (syn0). Index of the start of the
     *      weights for the current input word.
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
          // (I think this is the gradient calculation?)
          // Accumulate these gradients over all of the negative samples.
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          
          // Update the output layer weights by multiplying the output error
          // by the hidden layer weights.
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Once the hidden layer gradients for all of the negative samples have
        // been accumulated, update the hidden layer weights.
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
  
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
