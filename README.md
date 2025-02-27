#### Multi-Class Text Classification with LSTM

This repo aims to provide an example of how a Recurrent Neural Network (RNN) like Long Short Term Memory (LSTM) 
architecture can be implemented using Tensorflow 2.

#### Requirements

This project was implemented in built-in Python 3 installation on Linux and all dependencies are in the `requirements.txt`.




#### Quick Run

```
# clone this repo
git clone git@github.com:Ceceu/RNN_Text_Classifier.git

# enter the project folder
cd RNN_Text_Classifier

# instantiate a virtual env (make sure python3-venv is availabe on you system)
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install requirements
pip3 install -r requirements.txt

# run on a simple dataset using sample configs (sample_configs.json)
python3 app.py
```

#### Some Outputs


Last epochs of training (EarlyStopping callback finished the training before the maximum number of epochs):

```
54080/54112 [============================>.] - ETA: 0s - loss: 0.4246 - accuracy: 0.8748
Epoch 00026: val_accuracy did not improve from 0.83087
54112/54112 [==============================] - 223s 4ms/sample - loss: 0.4246 - accuracy: 0.8748 - val_loss: 0.5943 - val_accuracy: 0.8289
Epoch 27/50
54080/54112 [============================>.] - ETA: 0s - loss: 0.4129 - accuracy: 0.8793
Epoch 00027: val_accuracy did not improve from 0.83087
54112/54112 [==============================] - 222s 4ms/sample - loss: 0.4128 - accuracy: 0.8793 - val_loss: 0.6079 - val_accuracy: 0.8270
Epoch 28/50
54080/54112 [============================>.] - ETA: 0s - loss: 0.3992 - accuracy: 0.8838
Epoch 00028: val_accuracy did not improve from 0.83087
54112/54112 [==============================] - 224s 4ms/sample - loss: 0.3993 - accuracy: 0.8838 - val_loss: 0.5985 - val_accuracy: 0.8292
Epoch 29/50
54080/54112 [============================>.] - ETA: 0s - loss: 0.3870 - accuracy: 0.8876
Epoch 00029: val_accuracy improved from 0.83087 to 0.83170, saving model to model_checkpoints/best_rnn_text_classifier_model.hdf5
54112/54112 [==============================] - 225s 4ms/sample - loss: 0.3870 - accuracy: 0.8876 - val_loss: 0.5942 - val_accuracy: 0.8317

```
*Testing Accuracy*
```
6681/1 [================================================================] - 11s 2ms/sample - loss: 0.7690 - accuracy: 0.8228
Loss: 0.626 - Accuracy: 0.823
```
*Predicting*
```
new_text = [
        "I am a victim of identity theft and someone stole my identity and personal information to open up a Visa "
        "credit card account with Bank of America. The following Bank of America Visa credit card account do not "
        "belong to me : XXXX."]

```
Class Predicted: `Credit card`


