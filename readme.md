# Dataset from Kaggle 

## Scope:
Fraud risk is everywhere, but for companies that advertise online, click fraud can happen at an 
overwhelming volume, resulting in misleading click data and wasted money. Ad channels can drive 
up costs by simply clicking on the ad at a large scale. With over 1 billion smart mobile devices 
in active use every month, China is the largest
mobile market in the world and therefore suffers from huge volumes of fraudulent traffic.

TalkingData, China’s largest independent big data service platform, covers over 70% of active 
mobile devices nationwide. They handle 3 billion clicks per day, of which 90% are potentially 
fraudulent. Their current approach to prevent click fraud for app developers is to measure the 
journey of a user’s click across their portfolio, and flag IP addresses who produce lots of 
clicks, but never end up installing apps. With this information, they've built an IP blacklist 
and device blacklist.

## Methodology:
* Download Data from Kaggle
* Drop into Amazon RDS instance
* Feature Engineer  
* Goal - likelyhood an IP should be banned

### Potential additional step:
* predict clicks on a the most granular basis.

## Data Sources:
Kaggle TalkingData (Dataset Link)[https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/overview]

## Target:
MVP: Prediction of 
Goal: Integrate outside data sources to add further insight to values

## Features:
* ip: ip address of click. (encoded)
* app: app id for marketing. (encoded)
* device: device type id of user mobile phone (encoded)
* os: os version id of user mobile phone (encoded)
* channel: channel id of mobile ad publisher (encoded)
* click_time: timestamp of click (UTC)

## Target:
* attributed_time: if user download the app for after clicking an ad, this is the time of the app download (UTC)
* is_attributed: the target that is to be predicted, indicating the app was downloaded (0 or 1)


Things to consider:
There are 184mm rows and only 456,846 app downloads. this is like picking very small needles out of a very 
large hay stack. Need to find a way to make this scalable by getting random samples with the same proportion
of positives ~.24%.


# MVP:
Sampled 100,000 clicks from 150 unique ip addresses out of a total of
184,903,890 clicks from the original set. Looking to add more points to the train/test set
to improve f1. After which I would like to work on moving to a SVM
## Color commentary:
So far it looks as if MultinominalNB is leading the pack with an f1 score of .24

```
Performing Train & Tests...

Working on Baseline Dummy Classifier

                  precision    recall  f1-score   support
               0       1.00      1.00      1.00     49923
               1       0.00      0.00      0.00        77
        accuracy                           1.00     50000
       macro avg       0.50      0.50      0.50     50000
    weighted avg       1.00      1.00      1.00     50000

Working on Gaussian Naïve Bayes

                  precision    recall  f1-score   support
               0       0.64      1.00      0.78     32114
               1       0.66      0.00      0.01     17886
        accuracy                           0.64     50000
       macro avg       0.65      0.50      0.39     50000
    weighted avg       0.65      0.64      0.50     50000

Working on Bernoulli Naïve Bayes

                  precision    recall  f1-score   support
               0       1.00      1.00      1.00     49739
               1       0.44      0.15      0.22       261
        accuracy                           0.99     50000
       macro avg       0.72      0.57      0.61     50000
    weighted avg       0.99      0.99      0.99     50000

Working on Multinomial Naïve Bayes

                  precision    recall  f1-score   support
               0       0.99      1.00      1.00     49688
               1       0.56      0.16      0.24       312
        accuracy                           0.99     50000
       macro avg       0.78      0.58      0.62     50000
    weighted avg       0.99      0.99      0.99     50000
```
