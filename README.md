# CSE616 project - Image Captioning

By: Mahmoud Nada

Code: 2002387

Image Captioning is the process of generating a textual description for given images. It has been a very important and fundamental task in the Deep Learning domain. Image captioning has a huge amount of applications. NVIDIA is using image captioning technologies to create an application to help people who have low or no eyesight.
Several approaches have been made to solve the task. One of the most notable works has been put forward by Andrej Karpathy, Director of AI, Tesla in his Ph.D. at Standford.

Image captioning can be regarded as an end-to-end Sequence to Sequence problem, as it converts images, which are regarded as a sequence of pixels to a sequence of words. For this purpose, we need to process both the language or statements and the images, the result should be as the figure below.
![alt text](https://github.com/Mahmoud3211/CSE616_image_captioning/blob/master/images/ic.png)
## Installation

you can install requirements using 

```bash
pip install -r requirements.txt
```

## Usage
to use my trained model you need to download the checkpoint in this link:

https://drive.google.com/file/d/1Mm7PSXU_8UquzIMiyYXBTgFooJV8-Z8a/view?usp=sharing

extract the contents and place them in the saved model within the same directory of the repo

then you can use prediction.py to get a sample of predictions done by the model, or you can use evaluation.py to get the BLEU score of the model.


to train the model from scratch run the main.py file.

the test file is used to make sure that if the BLEU score code is working well by using the same captions in input and output if the result is not 1.0 then the code is not working well.