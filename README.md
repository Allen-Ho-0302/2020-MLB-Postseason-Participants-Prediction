Introduction:

See how stats like PA, AB, H, 2B, 3B, HR, RBI, SB, BA, OBP, SLG, etc influence postseason birth

Methods:

Gathering MLB regular season team stats from 2012-2019, including stats like PA, AB, H, 2B, 3B, HR, RBI, SB, BA, OBP, SLG, etc, 

Using SQL Server and Python(Spyder)

Building classification model

Try to build perfect model by comparing different learning rate, Model validation, early stopping, experiment different number of nodes in each layer, different number of layers

Results:

1.First Classification Model:

Epoch 1/1
240/240 - loss: 3920516.7883 - accuracy: 0.5875

2.Comparison of different learning rate:

Testing model with learning rate: 0.000001
Epoch 1/1
240/240 - loss: 164.5059
Testing model with learning rate: 0.010000
Epoch 1/1
240/240 - loss: 11612752.8642
Testing model with learning rate: 1.000000
Epoch 1/1
240/240 - loss: 8845915249735.2910

3.Model Validation:

Train on 168 samples, validate on 72 samples
Epoch 1/1
168/168 [==============================] - 0s 2ms/step - loss: 139.7968 - accuracy: 0.5119 - val_loss: 0.0000e+00 - val_accuracy: 1.0000

4.Early Stopping and Model Validation
Train on 168 samples, validate on 72 samples
Epoch 1/30
168/168 - loss: 377.0775 - accuracy: 0.4762 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 2/30
168/168 - loss: 146.2218 - accuracy: 0.5595 - val_loss: 262.4193 - val_accuracy: 0.0000e+00
Epoch 3/30
168/168 - 0s 93us/step - loss: 149.9949 - accuracy: 0.4762 - val_loss: 1.6557e-09 - val_accuracy: 1.0000


5.Experiment different number of nodes in each layer: result as png file

6.Experiment different number of layers: result as png file
