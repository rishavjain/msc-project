# Dependency Language Model for Lexical Substitution

The code and scripts in this repository were developed for the dissertation on dependency based lexical substitution. The code has been optimized to work as scheduled jobs on _iceberg_ platform (University of Sheffield HPC cluster).

The steps to run and evaluate the language model developed is given in the following steps:

1. Process the ukWaC data.
    + Parameters: (hard-coded)
        - Input file (file type: _gz_)
        - Output Path
        - No. of lines in output file (enables splitting the corpus data into multiple files)
    + Output:
        - Output files (text format)
        - List of all the output files created

    ```
    python preprocess_ukwac.py
    ```

2. Dependency Parsing.
    + Requirements:
        - Stanford CoreNLP. [http://stanfordnlp.github.io/CoreNLP]
    + Parameters: (hard-coded)
        - List of all input files
        - Path to Stanford CoreNLP package
        - Output Path
    + Output:
        - _iceberg_: schedules dependency parsing jobs for each input file and creates a file consisting of all the commands generated.
        - Windows: runs the dependency parser and generates the parsed output files.
    ```
    python dep_parse.py
    ```

3. Create word vocabulary from dependency parsed file (CONLL format).
    + Parameters:
        - Threshold (minimum count of word to consider)
        - Input file (file type: gz)
        - Output file
    + Output:
        - Word vocabulary where each line consists of a word and its count in the input data.
    ```
    python vocab.py <threshold> <input-file> <output-file>
    ```

4. Create context vocabulary.
    + Parameters: (paths are hard-coded)
        - Path to scripts
        - Path to dependency parsed file (file type: gz)
        - Output path
        - Threshold (minimum count of context to consider)
        - Window size for contexts
    + Output:
        - Context vocabulary where each line consists of a word and its context.
    ```
    ./run-cvocab.bash <threshold> <window-size>
    ```

5. Run _word2vecf_.
    + Parameters:
        - Path to scripts (hard-coded)
        - Context vocabulary file
        - Output path
        - Threshold (for any further filtering, if required)
        - Negative sampling
        - Embedding dimensions
    + Output:
        - Word and context embeddings (_numpy_ readable format)
    ```
    ./run-word2vec.bash <context> <model-out> <threshold> <neg-samp> <dim>
    ```

6. Find substitutes and evaluate the model.
    + Requirements:
        - WORDNET corpus in NLTK python package
    + Parameters:
        - Path to scripts (hard-coded)
        - Path to word and context embeddings
        - Output path
        - Measure to use for substitution
        - Dataset to use (LST or CIC)
        - Window size for contexts
    ```
    ./run-evaluate.bash <emb-path> <out-path> <measure> <dataset> <window-size>
    ```
