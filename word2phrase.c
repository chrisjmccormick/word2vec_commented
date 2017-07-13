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

#define MAX_STRING 60

const int vocab_hash_size = 500000000; // Maximum 500M entries in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  char *word;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
struct vocab_word *vocab;
int debug_mode = 2, min_count = 5, *vocab_hash, min_reduce = 1;
long long vocab_max_size = 10000, vocab_size = 0;

// The total number of words in the training corpus, tallied in the 
// "LearnVocabFromTrainFile" function.
long long train_words = 0;

real threshold = 100;

unsigned long long next_random = 1;

/**
 * ======== ReadWord ========
 * Reads a single word from a file, assuming space + tab + EOL to be word 
 * boundaries.
 *
 * NOTE: This function is identical with ReadWord in word2vec.c
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
        // Put the newline back so that we read it next time.
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
  unsigned long long a, hash = 1;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
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
    vocab_max_size += 10000;
    vocab=(struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }

  // Add the word to the 'vocab_hash' table so that we can map quickly from the
  // string to its vocab_word structure. 
  
  // Hash the word to an integer between 0 and 30E6.  
  hash = GetWordHash(word);
  
  // If the spot is already taken in the hash table, find the next empty spot.
  while (vocab_hash[hash] != -1) 
    hash = (hash + 1) % vocab_hash_size;
    
  // Map the hash code to the index of the word in the 'vocab' array.    
  vocab_hash[hash]=vocab_size - 1;
  
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
 * NOTE: This function is identical to SortVocab in word2vec.c, but doesn't
 *       include the memory allocation for the binary tree.
 * 
 * Removing words from the vocabulary requires recomputing the hash table.
 */
void SortVocab() {
  int a;
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
  
  // For every word currently in the vocab...
  for (a = 0; a < vocab_size; a++) {
    
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      // Decrease the size of the new vocabulary.
      vocab_size--;
      
      // Free the memory associated with the word string.
      free(vocab[vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
    }
  }

  // Reallocate the vocab array, chopping off all of the low-frequency words at
  // the end of the table.  
  vocab = (struct vocab_word *)realloc(vocab, vocab_size * sizeof(struct vocab_word));
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
  char word[MAX_STRING], last_word[MAX_STRING], bigram_word[MAX_STRING * 2];
  FILE *fin;
  long long a, i, start = 1;
  
  // Initialize the hash table to -1.
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  
  // Open the training text file.
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
    
    // If word is the token </s> then set start = 1 and skip it.
    if (!strcmp(word, "</s>")) {
      start = 1;
      continue;
    } else start = 0;
    
    // Count the total number of words in the training corpus.
    train_words++;
    
    // Print progress at every 100,000 words.
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("Words processed: %lldK     Vocab size: %lldK  %c", train_words / 1000, vocab_size / 1000, 13);
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
    
    // TODO - This shouldn't be reachable...
    if (start) continue;
    
    // Combine the previous word with the current one:
    //    bigram_word = last_word + "_" + word
    sprintf(bigram_word, "%s_%s", last_word, word);
    
    // Add a null character to the last possible position.
    bigram_word[MAX_STRING - 1] = 0;
    
    // Set last_word = word.
    strcpy(last_word, word);
    
    // Lookup the combined word to see if we've already added it.
    i = SearchVocab(bigram_word);
    
    // If not, add it to the vocabulary.
    if (i == -1) {
      a = AddWordToVocab(bigram_word);
      vocab[a].cn = 1;
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
    printf("\nVocab size (unigrams + bigrams): %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  
  fclose(fin);
}

/**
 * ======== TrainModel ========
 * Main body of this tool.
 *
 * This function performs the phrase detection and simultaneously generates a 
 * new training file which contains the phrases replaced with underscores 
 * between the words. For example, "New York" becomes "New_York" in
 * the new file.
 *
 * Phrase detection occurs in two steps.
 *
 * STEP 1: Learn a vocabulary
 *   In this step, we're just identifying all the unique words in the training
 *   set and counting the number of times they occur. We also add to the 
 *   vocabulary every combination of two words observed in the text. For 
 *   example, if we have a sentence "I love pizza", then we add vocabulary 
 *   entries and counts for "I", "love", "pizza", "I_love", and "love_pizza".
 *
 *
 * STEP 2: Decide which word combinations represent phrases.
 *   In this step, we go back through the training text again, and evaluate
 *   whether each word combination should be turned into a phrase.
 *   We are trying to determine whether words A and B should be turned into 
 *   A_B. 
 *
 *   The variable 'pa' is the word count for word A, and 'pb' is the count for
 *   word B. 'pab' is the word count for A_B. 
 *
 *   Consider the following ratio: 
 *     pab / (pa * pb)
 *   
 *   This ratio must be a fraction, because pab <= pa and pab <= pb.
 *   The fraction will be larger if:
 *     - pab is large relative to pa and pb, meaning that when A and B occur 
 *       they are likely to occur together.
 *     - pa and/or pb are small, meaning that words A and B are relatively 
 *       infrequent.
 *   
 *   They modify this ratio slightly by subtracting the "min_count" parameter
 *   from pab. This will eliminate very infrequent phrases. The new ratio is
 *     (pab - min_count) / (pa * pb)
 *   
 *   Finally, this ratio is multiplied by the total number of words in the 
 *   training text. Presumably, this has the effect of making the threshold
 *   value more independent of the training set size.
 */
void TrainModel() {
  
  //  pa - The word count (number of times it appears in the training corpus)
  //       of the previous word.
  //  pb - The word count of the current word.
  // pab - The word count of the combined word (previous_current)
  // oov - A flag, set to 1 to if any of either the previous, current, or 
  //       combined words is not in the vocabulary.
  //   i - The index into the vocab of the current word.
  //  li - The index into the vocab of the previous word.
  //  cn - A running count of the number of training words.
  long long pa = 0, pb = 0, pab = 0, oov, i, li = -1, cn = 0;
  
  char word[MAX_STRING], last_word[MAX_STRING], bigram_word[MAX_STRING * 2];
  
  real score;
  
  FILE *fo, *fin;
  
  printf("Starting training using file %s\n", train_file);
  
  // Build the vocabulary from the training text.
  // 
  // There will be a vocabulary entry for every word, as well as every 
  // combination of two words that appear together. The only exception is
  // that the least common words and phrases will be removed if the vocabulary
  // grows too large.
  LearnVocabFromTrainFile();
  
  // The training file was opened, read, and closed in the previous step.
  // Now we need to open the training file and the output file.
  fin = fopen(train_file, "rb");
  fo = fopen(output_file, "wb");
  
  word[0] = 0;
  
  while (1) {
    
    // Update the 'last_word' string with the word from the previous iteration.
    // last_word is word A.
    strcpy(last_word, word);
    
    // Read the next word (word B) from the training file.
    ReadWord(word, fin);
    
    // Check for the end of the training file.
    if (feof(fin)) 
      break;
    
    // If the word is the </s> token, then just write a newline and continue
    // to the next word.
    if (!strcmp(word, "</s>")) {
      fprintf(fo, "\n");
      continue;
    }
    
    // Count the number of words in the training file.
    cn++;
    
    // Print progress update every 100,000 input words.
    if ((debug_mode > 1) && (cn % 100000 == 0)) {
      printf("Words written: %lldK%c", cn / 1000, 13);
      fflush(stdout);
    }
    
    // If this flag becomes 1, then we won't combine A and B.
    oov = 0;
    
    // Lookup the current training word (word B).
    i = SearchVocab(word);
    
    // If B isn't in the vocabulary, don't combine A and B.
    if (i == -1) 
      oov = 1; 
    // Otherwise, lookup the word's frequency and store it in 'pb'.
    else 
      pb = vocab[i].cn;
    
    // If word A wasn't in the vocab, then don't combine A and B.
    if (li == -1) 
      oov = 1; 
      
    // Track the index of the previous word.   
    li = i;
    
    // Combine the previous and current words:
    //   bigram_word = last_word + "_" + word
    sprintf(bigram_word, "%s_%s", last_word, word);
    bigram_word[MAX_STRING - 1] = 0;
    
    // Lookup the combined word (A_B).
    i = SearchVocab(bigram_word);
    
    // If the combined word isn't in the index, don't write A_B.
    if (i == -1) 
      oov = 1;
    // Otherwise, lookup the count for the combined word and store it in 'pab'.
    else 
      pab = vocab[i].cn;
    
    // Don't combine the words if either word A or word B occur fewer than 
    // min_count (default = 5) times in the training text.
    if (pa < min_count) oov = 1;
    if (pb < min_count) oov = 1;
    
    // Calculate a score for the word combination A_B.
    //
    // The scoring function is described in the function header comment.
    //
    // The score is higher when:
    //   - A and B occur together often relative to their individual 
    //     occurrences.
    //   - A and B are relatively infrequent.
    //
    // The score is zero if A_B isn't in the vocabulary.
    if (oov) 
      score = 0; 
    else 
      score = (pab - min_count) / (real)pa / (real)pb * (real)train_words;
    
    // If the score for A_B is high enough, write out "_B"    
    if (score > threshold) {
      fprintf(fo, "_%s", word);
      pb = 0;
    // Otherwise, write out " B" to keep A and B separate.
    } else fprintf(fo, " %s", word);
    
    // Word B will be the "previous" word on the next iteration, so replace 
    // the word count for A with the count for B.
    pa = pb;
  }
  
  fclose(fo);
  fclose(fin);
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
    printf("WORD2PHRASE tool v0.1a\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters / phrases\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-threshold <float>\n");
    printf("\t\t The <float> value represents threshold for forming the phrases (higher means less phrases); default 100\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\nExamples:\n");
    printf("./word2phrase -train text.txt -output phrases.txt -threshold 100 -debug 2\n\n");
    return 0;
  }
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threshold", argc, argv)) > 0) threshold = atof(argv[i + 1]);
  
  // Allocate the Vocabulary - TODO...
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  
  // Allocate the vocabulary hash - TODO...
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  
  // Run phrase detection.
  TrainModel();
  
  return 0;
}
