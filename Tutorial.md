TRAIN:

The database is created first, using the python code 'database.py'. 

To train the model, I used the GPU on google colab as it is quicker. This means uploading the database .h5 file, along with both json files and all the training audio data. The code for this is given in 'ResNet_classifier.py'. 

TEST:

The code for creating the detector is the file 'Detector.py'. You need to download the model (.kt file) if you used google colab before. This code will run the model across the test files to create a set of detections and a comparison file for how they compare to the annotations file. There are plenty of parameters to adjust in the process method, all of which are explained on the ketos website. 

[https://docs.meridian.cs.dal.ca/ketos/modules/neural](https://docs.meridian.cs.dal.ca/ketos/modules/neural_networks/dev_utils/detection.html?highlight=threshold)*[networks/dev](https://docs.meridian.cs.dal.ca/ketos/modules/neural_networks/dev_utils/detection.html?highlight=threshold)*[utils/detection.html?highlight=threshold](https://docs.meridian.cs.dal.ca/ketos/modules/neural_networks/dev_utils/detection.html?highlight=threshold)

RESULTS:

I haven't augmented any training data yet so there is plenty of room to improve there, however, the model I've put here achieved a val_loss of 0.152 after 30 epochs.

Using the exact parameters in the files, the model detected 86% of the 1246 calls, creating 108 false positives. 

One issue I am having is that increasing the batch size in the process method shouldn't affect the results, only decrease the time it takes. When I increase it however I end up with slightly fewer detections.

I need to have a closer look at these results as the duration of some of the calls is as long as 30 seconds, but the majority look fairly consistent with the spectrograms.

As you can see in the spec_config file I have changed the spectrogram frequency range to 10Hz-50Hz which appears to have increased the accuracy for now. 