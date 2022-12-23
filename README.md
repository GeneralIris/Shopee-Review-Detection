<!-- TOC -->
# Table Content
- [Shopee Review Detection](#spam-detection-system-on-shopee-review)
  - [Why?](#why)
  - [Dataset Info](#dataset)
  - [Obtaining Dataset](#this-is-da-police-where-you-get-the-data)
  - [Models Used](#models)
  - [How to use?](#usage)
  - [Specifications](#specifications)
  - [Repo Notes](#repo-notes)
  - [Credits](#credits)
- [Games I like](#games-i-like)

<!-- /TOC -->

# Spam Detection System on Shopee Review

## Why?
I hate current Shopee reviews. So I decide to make a spam detection system for it.

* It's a machine learning driven towards detecting spam reviews on any Shopee product for my FYP that focuses on Storage Devices product. 
* A system integrated using flask with mysql as the database used to simulate the user access to the model.

**It works for non-storage devices, huh I wonder ;)** 


## Dataset
### This is da police! Where you get the data?
* Obviously from Shopee itself using their API 
* Labels? I did it myself with my own set of rules to be considered as Spam / Ham

### Total dataset used?
- 500+ Ham
- 600+ Spam <br>
- *Original Spam Count Not Included= 7k+ Kill me already*


## Models
### What are the machine learning used?
* Random Forest
* Support Vector Machine
* Naive Bayes


## Usage
### How to use the system?
* A user may input the URL into the system, using Shopee API in return, it spits out all the 'ham' reviews with a cooldown. As each 3 days, if not a single user made the search on the system, it will perform the detection otherwise less than 3 days, it will instead grab the data from the database.
* Well, this to avoid high traffic grabbing data from the store and time to process duplicates review.
**Might try improve this**
**Update: Its better**

### With Database
- You need to config a db on phpMyAdmin (I used) with all the stored proc I built.
**Not sure if I include them or not**


## Specifications
### What's your machine specs during the development?
* Ryzen 5600X @ 4.4GHZ all core
* ASUS TUF RTX3080
* 16GB 3600mhz
* 2 SSDs and 3 HDDs (Project on SSD)

### Training
* Yes, but only on CUDA supported devices. 
* Python 3.9. Any thing above will get messy.
* In terms of dependencies I really forgot to update them. But just install the packages in the notebook.


## Repo Notes
### Why is it too simple?
- Well this is my final year project, currently just finished preparing the machine learning + 3 models (RF,SVM,NB) used for evaluating the best among them. <br>
- For now, I share my model train and testing + the jupyter notebook as the prove of my work, 'Working on finalizing my research paper ;)'
**Ps~ Its SVM ;)**

## Credits
Thanks to :
- MalayNLP for allowing such an awesome tool to manage Malay/Manglish text
- Other researchers & lecturers that guide me on NLP and loading models


## Games I like
Horizon Zero Dawn, Ori and Metal HellSinger

