# Sentiment Analysis with GRUs

A deep learning approach to emotion classification in text using Gated Recurrent Units (GRUs). This model achieves 93% accuracy in classifying tweets into six emotional categories: sadness, joy, love, anger, fear, and surprise.

## Overview
This project implements a GRU-based neural network for sentiment analysis, demonstrating significant improvements over traditional RNN approaches. The model processes tweet text through word embeddings and GRU layers to capture complex emotional patterns in natural language.

## Key Features
- Multi-class emotion classification (6 categories)
- GRU-based architecture with word embeddings
- 93% classification accuracy
- Comprehensive visualization of model performance
- Analysis of semantic learning through embedding comparisons

## Model Architechture
1. Input layer (tokenized text sequenxes)
2. Embedding layer (dense vector representations)
3. GRU layer (128 units, tanh activation)
4. Dense layer (softmax activation for classification)

## Dataset
Source: Kaggle Emotions dataset
Size: 70,000 labeled tweets
Training split: 60,000 examples
Testing split: 10,000 examples
Categories: sadness, joy, love, anger, fear, surprise

## Basic requirements
python>=3.8
tensorflow>=2.0
keras
numpy
pandas
matplotlib
scikit-learn

## Installation
```git clone https://github.com/branley1/predicting-emotions-with-grus```
```cd predicting-emotions-with-grus```

To access our data, uncompress the attached zip file ```archive.zip``` into a ```text.csv``` file.

To access our project, run the following commands:

```
source /usr/swat/bin/CS63env
jupyter-lab
```

Once you are in the jupyter page, open the file named ```RNN.ipynb```

There, each code block has comments that describe what that code block does. Replace the variable ```filename``` with the path to the ```text.csv``` file. Press run all to run all code blocks. Alternatively, select a specific block and press Shift+Enter to run that specific block.

## Key Findings
- Strong performance in distinguishing between most emotion categories
- Some confusion between closely related emotions (joy/love)
- Effective semantic learning demonstrated through embedding analysis
- No significant overfitting observed in learning curves

## Future Improvements
- Experiment with longer text samples beyond tweets
- Implement LSTM and Bi-directional LSTM comparisons
- Expand emotion categories
- Add real-time prediction capabilities

## Contributors
1. Branley Mmasi
2. Alex Huynh

## Acknowledgements
- Based on initial implementation from GeeksforGeeks' RNN sentiment analysis
- Dataset provided by Nidula Elgiriyewithana on Kaggle
- Created as part of CS63 Artificial Intelligence (Spring 2024) at Swarthmore College.

## License
This project is licensed under the MIT License — see the LICENSE file for details.
