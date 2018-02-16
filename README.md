# Skype-Text-Generator

## Context
As mentioned in the repository description, I've been playing around with the idea of creating a text generator based on a Skype chat group. Initially, the goal of this project was to jokingly create an artifical replacement for one of our friends who's notorious for ignoring messages! I got a hold of the data (omitted from the repository for privacy reasons) a few months ago and tried some basic text generation at a character level. The idea here is that every sequence of k characters can be used to predict the k+1th character thus forming a supervised learning task. This didn't end up working too well... perhaps due to insufficient data (I was trying to model one person) or because of the model architecture (adapted from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/). I then started investigating the alternative of making a chatbot instead using a sequence-to-sequence model instead but soon realized I didn't have nearly enough data to create something meaningful with the personality of one of our group members. So, ultimately, I shelved the project.

Although I don't have any particular rush on this project, it's an interesting way to play around with a different subset of NLP tasks.

## Goals
I recently returned to this project but this time with the goal of using the entire Skype chat history rather than just one person. There still isn't enough data for a chatbot, so I figured I'd start with text generation. In terms of preprocessing, I've performed basic string cleaning to each message and just grouped everything together such that it is one long string. I'm trying to accomplish the following:
1. Test the simple model using the larger (entire chat) dataset and qualitatively observe performance with a basic character sequence learning LSTM.
2. Try playing around with character sequence length, using words instead of characters, and model width/depth
3. Change data preprocessing to create meaningful divisions between different people talking. Right now it's basically learning a unified and uninterruped stream of messages.
4. If it works well, consider transforming it into a Discord bot.

Update 2/8/2018:
- Testing with [Shapespeare dataset](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) before applying to skype chats as the former has more structure and is thus easier to learn.
- Changed data representation to vectorized form as per the [Keras documentation](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py). Essentially this is a one-hot-encoded version of the previous ordinal character strategy.
- Replaced argmax sampling on predictions with argmax on a multinomial experiment of predictions. This allows for greater creativity in the model.
- Added temperature/disparity. This scales the predictions allowing control over model creativity. Low temperature results in more logical sentences but less creativity.
- The RMSProp optimizer doesn't seem to converge; replaced with Adam.
- Implemented the on_epoch_end method provided in the Keras text generation documentation to print a generated sample at various disparities at the end of each epoch. This allows one to keep track of progress.

The model seems to work well with the Shakespeare text file and generates some fairly good predictions even with only 15 epochs. I suspect this is due, in part, to Shakespeare's writing style and the structure in his work. Comparatively, the Skype logs do not show as much improvement within the first few epochs. The model is still training so we'll see how it performs at the end of its initial 20 epoch run. A meaningful way to improve it would be to consider artificially introducing some structure in the data. For example, divide between the people talking (perhaps present it as a play) and add commas between consecutive messages.

Update 2/10/2018:
- After about 8 iterations, the training loss (for the Skype chats) doesn't show much improvement and the model doesn't qualitatively generate well. As previously mentioned, this is likely due to the lack of structure in the training data. I'll investigate how to tackle that.
- Out of interest, I'll add an examples of Shakespeare generated text (at various disparities) to the examples folder after training the model to a cross entropy loss of below 1.0. I'll also try running this script on lyrical data to see how it works.
    - added shakespeare example

## Potential Future Steps
- Read some papers on state-of-the-art in text generation tasks and apply those here.
- Perhaps apply the model to the lyrical discographies of musical artists. This particular task is fairly well-known, but it would be interesting to investigate it anyways given that I already have a script for webscraping lyrics (my VisualizingSongLyrics repository).
