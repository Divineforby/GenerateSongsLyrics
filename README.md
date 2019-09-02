# GenerateSongsLyrics

This project aimed to use an LSTM based recurrent neural network to generate new song lyrics given a corpus of popular pop songs using the Pytorch library.

Data can be found at this kaggle link: https://www.kaggle.com/mousehead/songlyrics

Using a character based approach to language modeling we train the LSTM network on a variety of pop songs. The model receives a one-hot encoded character at and asked to output a probability distribution for the next character in the sequence. 

To generate new text we feed the model the SOS(Start of Sentence) character and sample the output for the next character, we then recursively feed this sample to the model and repeat. The process is finished once an EOS(End of Sentence) is reached.

If the model has sufficiently understood the language model, proven by the validation loss decreasing, we should expect to see coherent words and grammar as well as structure of the text. 

In an effort to allow further fine-tuning of the text generation we introduce the temperature hyperparameter.
Given that the distribution the model outputs is defined using a Softmax function on the raw real-valued outputs of the model we can change the bias of this distribution by dividing those outputs by the temperature.

In effect, as temperature increases we see more uniformity in the distribution and the sampling process becomes more random and the output more garbled. Vice versa, as temperature decreases minute differences in probabilities get magnified and thus even slightly more favored output values are much more likely and therefore we see a more deterministic and coherent sample.

We try to find a middle-ground between the garbled mess output and the completely deterministic unoriginal output.

Python Files:

train_model.py - Responsible for the entire training process of a new model.

data_loader.py - Contains functions and classes pertaining to the management of our data.

models.py - Defines the custom neural network class we will implement and train.

configs.py - Holds configurations for our model and the training/generation process.


How to:

Train -

Command: python train_model.py

The code will create a new model and encapsulate the entire training process under model_checkpoints/train_sessionX. Where X is the last training session index + 1. All configurations for the model and the training procedure can be seen and configured in configs.py.

Checkpoints for the model and optimizer will be saved in this folder.
There will also be a log for printed values of training and validation loss as well as pickle files of the final loss arrays.

Generate -

Command: python generate.py <Temperature> <Number of Samples> <Max Length of Samples>
    
Takes in 3 arguments to begin generating new song lyrics. All output will be in the results folder.
Path to which model to use can be defined in the file itself in the main function.


Default Files:

By default model_checkpoints will contain a training session with a final model. This model is one that I've trained before.
The validation loss for this model is ~0.55. Although this model does a decent job at text generation better results expected with longer training time.

There are a few files in results that demonstrate the models abilities to generate text at different Temperature. I found that 0.7 - 0.8 is the sweet spot for variety and coherence in text generation.





