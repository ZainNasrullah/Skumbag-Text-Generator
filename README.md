# Skumbag-Text-Generator

## Context
As mentioned in the repository description, I've been playing around with the idea of creating a text generator based on a Skype chat group. Initially, the goal of this project was to jokingly create an artifical replacement for one of our friends who's notorious for ignoring messages! I got a hold of the data (omitted from the repository for privacy reasons) a few months ago and tried some basic text generation at a character level. The idea here is that every sequence of k characters can be used to predict the k+1th character thus forming a supervised learning task. This didn't end up working too well... perhaps due to insufficient data (I was trying to model one person) or because of the model architecture (adapted from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/). I then started investigating the alternative of making a chatbot instead using a sequence-to-sequence model instead but soon realized I didn't have nearly enough data to create something meaningful with the personality of one of our group members. So, ultimately, I shelved the project. 

Although I don't have any particular rush on this project, it's an interesting way to play around with a different subset of NLP tasks. 

## Current Goals
I recently returned to this project but this time with the goal of using the entire Skype chat history rather than just one person. There still isn't enough data for a chatbot, so I figured I'd start with text generation. In terms of preprocessing, I've performed basic string cleaning to each message and just grouped everything together such that it is one long string. I'm trying to accomplish the following:
1. Test the simple model using the larger (entire chat) dataset and qualitatively observe performance with a basic character sequence learning LSTM.
2. Try playing around with character sequence length, using words instead of characters, and model width/depth
3. Change data preprocessing to create meaningful divisions between different people talking. Right now it's basically learning a unified and uninterruped stream of messages. 
4. If it works well, consider transforming it into a Discord bot. 

## Next Steps
- Read some papers on state-of-the-art in text generation tasks and apply those here.
- Perhaps apply the model to the lyrical discographies of musical artists. This particular task is fairly well-known, but it would be interesting to investigate it anyways given that I already have a script for webscraping lyrics (my VisualizingSongLyrics repository). 
