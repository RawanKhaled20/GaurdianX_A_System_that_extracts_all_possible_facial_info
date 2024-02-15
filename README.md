GuardianX is a smart system prototype designed using AI_based technology (Tensorflow and goolge colab and kaggle) to train multiple models to work on facial info extraction like (Drowsiness state through opened/closed eyes, emotional states (happy, sad, angry, surprised, neutral), age range (young, middel, old), gender (female or male), yawning or no yawning, masks on or not and lastly asks you on your feedback and analyze the sentimnets(positive, negative, neutral) to make the best out of people opinions.

This model can be used in humonoid robots to provide them with emotional intellegence or in smart driving systmes as a guide for you while driving.

To be able to Use this system fisrt you have to download all the .h5 models files for the face extraction and the .plk file for sentiments analysis , also the response files as they are alamrs produced for guding you.

secondly, save all the downlaoded models files to the working directory and copy the python code in the main.py to any python IDE (I used pycharm) to run the code and make sure you pip install all the following libraries (tensorflow, opencv-python, numpy, matplotlib, ultralytics, pygame, tkinter, nltk, cvzone, scikit-learn)

lastly, make sure to not open multiple tabs in the background as that will lag the system even more as it is very affected by the things running the OS

To access the used models training codes visit this links: https://github.com/RawanKhaled20/Sentimental-analysis.git, https://github.com/RawanKhaled20/YawningDetection.git, https://github.com/RawanKhaled20/GenderDetection.git, https://github.com/RawanKhaled20/AgeDetection.git
, https://github.com/RawanKhaled20/Emotion_extraction.git, https://github.com/RawanKhaled20/Drowsiness_Detection.git, https://github.com/RawanKhaled20/MaskDetection.git
