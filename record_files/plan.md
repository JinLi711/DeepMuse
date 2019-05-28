# Overview

1. Train a language model to generate music.
2. Rate a peice of text with emotional scores, and convert those scores into starting notes.
3. Generate the music with the model.
4. Apply methods to make the music sound better.


* C - Complete
* D - Doing
* NW - Not working on yet
* ND - Decided not to do this



# Step 1: Set Up

| Progress | Date Finished | Task                  
|----------|---------------|-----
|C         | 5/27/2019     | Set up the repository.
|C         | 5/28/2019     | Get a license.





# Step 1: Research 

| Progress | Date Finished | Task                  
|----------|---------------|-----
|C         | 5/11/2019     | Research how to map words to emotional ratings.
|C         | 5/12/2019     | Research how to convert emotional ratings into musical notes.
|C         | 5/22/2019     | Research how to represent MIDI timesteps as a sequence of vectors.
|C         | 5/23/2019     | Research the different language models that we can use for music generation.
|NW         | NA     | Research algorithms that help integrate music notes when key is changed.
|NW         | NA     | Research the psychology behind what makes music sound pleasant.
|NW         | NA     | Research how music generation is evaluated.
|NW         | NA     | Find pieces of literature that we want to convert to music.

# Step 2: Visualization

| Progress | Date Finished | Task                  
|----------|---------------|-----
| NW | NAN| Visualize the statistics of the literature that we are using. Example: emotional counts.


# Step 3: Preprocess

| Progress | Date Finished | Task                  
|----------|---------------|-----
|C         | 5/28/2019     | Split the text by sentences.
|C         | 5/27/2019     | For each sentence, normalize the text by lowercasing and removing all punctuation. 
|NW         | NAN     | Use a tokenizer to parse through text.
|C         | 5/28/2019     | Count occurences of each emotion for each sentence.
|C         | 5/28/2019     | Create a cumulative distribution of positive - negative counts of words.
|C         | 5/28/2019     | Smooth the cumulative distribution.
|C         | 5/28/2019     | Find local minimums and maximums of the cumulative distribution. Keep the points that indicate high changes.


# Step 4: Model Building

| Progress | Date Finished | Task                  
|----------|---------------|-----
|NW         | NAN     | For now, lets see if we can just a pretrained model from Google Magenta.



# Step 5: Model Evaluation

| Progress | Date Finished | Task                  
|----------|---------------|-----


# Step 6: Model Deployment

| Progress | Date Finished | Task                  
|----------|---------------|-----
| NW        | NAN     | Javascript visualization.


# Step 7: Final Adjustments 

| Progress | Date Finished | Task  
|----------|---------------|-----
| NW        | NAN     | Improve code documentations on preprocess files.
| NW        | NAN     | Update the README.md file.
| NW        | NAN     | Build documentation with Sphinx.
| NW        | NAN     | Create directory tree.
| NW        | NAN     | Remove warnings.
| NW        | NAN     | Create a Pypi package.
