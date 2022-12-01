# Why no repo?
Well this is my final year project, currently just finished preparing the machine learning + 3 models (RF,SVM,NB) used for evaluating the best among them. <br>
For now, I share my model train and testing + the jupyter notebook as the prove of my work, 'Working on finalizing my research paper ;)'
## Ps~ Its SVM ;)


# Spam Detection System on Shopee Review
- I hate current Shopee reviews. So I decide to make a detection spam system for it.


## What is this monstrosity!
It's a machine learning driven towards detecting spam reviews on any Shopee product for my FYP it focuses on Storage Devices.
A system integrated using flask with mysql as the database used to simulate the user access to the model.

## How to use the system?
- A user may input the URL into the system, using Shopee API in return, it spits out all the 'ham' reviews with a cooldown. As each 3 days, if not a single user made the search on the system, it will perform the detection otherwise less than 3 days, it will instead grab the data from the database.
- Well, this to avoid high traffic grabbing data from the store and time to process duplicates review. <br>
*Might try improve this*

## Total dataset used?  CurrentData.csv
- 500+ Ham
- 600+ Spam <br>

*Original Spam Count Not Included= 7k+*

## What is/are machine learning used?
- Random Forest
- Support Vector Machine
- Naive Bayes


## Can you train them?
- Yes, but only on CUDA supported devices. 
- Python 3.9. Any thing above will get messy.
- In terms of dependencies I really forgot to update them. But just install the packages in the notebook.

## To run the system?
- You need to config a db on phpMyAdmin (I used) with all the stored proc I built. <br>
*Not sure if I include them or not*

## What's your machine specs during the development?
- Ryzen 5600X @ 4.4GHZ all core
- ASUS TUF RTX3080
- 16GB 3600mhz
- 2 SSDs and 3 HDDs (Project on SSD)

# Games I like ;0
Horizon Zero Dawn, Ori and Metal HellSinger

