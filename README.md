
# Sentiment Analysis and Image Classification of Yelp Restaurant Reviews and Images in the Tampa Bay Area. 

This project aims to classify restaurant images as either fine dining or fast food using a combination of images associated with food, beverages, interior, exterior, and menus. In addition the sentiment of Yelp User Reviews are classified as either positive or negative.

BLIP: Bootstrapping Language-Image Pre-training transformers from HuggingFace were used on 5,807 images to extract text captioning from them. OpenAI CLIP is used for the image and text pairs to make a binary prediction. The returned output classifies the image as being either fine dining or fast food. 

DistilBERT is a fine-tuned model which is a smaller and faster version of BERT and produces an output which indicates whether input text is positve or negative. Training was completed on 474,385 individual Yelp restaurant reviews. The reviews from the dataset were filtered to 1 & 5 star reviews only. The output prediction of negative should correspond with 1 star ratings while positive should correspond with 5 star ratings. 

An accuracy of 99.5% was achieved for the Yelp Restaurant Reviews while a 93.9% accuracy was achieved with the Image Classifications. The deployment leverages PyTorch for the deep learning framework and Amazon SageMaker for model training, hosting, and monitoring, with a FastAPI backend for serving predictions.

This application could be used by potential restaurant owners to better understand how their restaurant is perceived by their photos and customer reviews as well as assist owners in positioning themselves in their desired market segment. This model could also be used by consumers seeking specific dining experiences and to aid in the discovery of new upscale dining options.

## Demo

Please follow link below to access the application. 

https://5cde-3-135-152-169.ngrok-free.app/

## Images
Refer to CSV file called "sample_new_images_captions.csv" to view text captions and labels associated with the sample images. 

These include images the model has never seen before and are associated with Yelp Images from different regions (i.e., California, Tennessee, & Louisiana). Please see "Sample Images" folder for 4 sample images that can be uploaded into the model. 

## Yelp Reviews
Refer to CSV file called "sample_new_yelp_reviews" to view 4 sample text reviews that the model has never seen before. Also associated with the regions mentioned above. Feel free to use these or even your own text reviews to view the sentiment of the reviews.

You could copy and paste one single review or a batch of reviews and it will return one or more predictions. If you use a batch of review as the text input, please be sure to copy and paste the text in a list format and enter a space between each review. 

### Here is an example: 
"Terrible! No customer service what so ever. We sat and waited for an order to be placed. They never answered with the button pushed. We said something to one of the car hops and said that one wasn't working.  No sign placed on there. He didn't offer to take our order or any help what do ever.",

"Yessss!! Finally a kbbq in New Orleans worth going to! You get a lot of meat on a plate for a good price. Everything was very flavorful! Went on grand opening night so they ran out of a few dishes that I wanted to try but I definitely will be coming back! Staff was nice and quick too."