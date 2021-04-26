# Moral_Foundation_FrameAxis
This library provides the code for a flexible calculation of moral foundation scores from textual input data.

## Command-Line Arguments
- **[--input_file]:** Path to the dataset .csv file containing input text documents in a column.
- **['--docs_colname']:** The name of the column in the input file that contains the texts to calculate the MF scores on. 
- **[--dict_type]:** Dictionary for calculating FrameAxis Scores. Possible values are: emfd, mfd, mfd2
- **[--word_embedding_model]:** Path to the word embedding model used to map words to a vector space. If not specified a default w2v model will be used.
- **[--output_file]:** The path for saving the MF scored output CSV file. The output file contains columns for MF scores concatenated to the original dataset.

### To-dos: 
- word-embedding updating tool.
- Supporting MFD2.
- Document TFIDF tool. 
- Add package requirements and documentation for how to setup python. 
