# NLP Demo: Trump Tweet Bot From Scratch

<p align="center">
<img align="center" src="https://raw.githubusercontent.com/EdenTrainor//natural-tweet-generation/master/resources/trump_wordcloud.PNG" alt="djt" width="250" height="250">
</p>

### A simple demo of some natural language processing. Staring twitter icon, Donald J. Trump


The demo includes a little exploratory data analysis as well as the creation of the above word cloud. 

We then go on to utilize some standard NLP techniques for processing and embedding data into neural network architecture.
The model used is a multi-layer LSTM with dropout; trained via teacher forcing.

The model is used to generate the probabilities of the next rune in the sequence after a seed phrase is supplied.
A "creativity" hyper-parameter is used to scale the distribution to allow the tweet bot a little more character in it's predictions.

This project  just for fun and I hope you enjoy having a glance over it. Positive and negative criticism is equally welcome, we're all here to learn!
The data source for this project is: https://data.world/briangriffey/trump-tweets, many thanks to Brian Griffey.
