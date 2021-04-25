# Linkedin-IA316-2020

## Context

This projet aims to produce a Linkedin lookalike environment for the job offer recommendation system.

Different agents have been implemented to interact with the environment.

## Environnement

The environment contains a specified amount of users and offers that contain several features such as the skills, the age, ... It can be an evolutive environment, meaning that the users and offers can be deleted and added during the execution, or stay the same throughout an experiment.

The reward associated to each action chosen by an agent is the probability for the user to click on the offer chosen by the algorithm. The calculous probability can be found in the code of the environment.

## Agents

Several agents have been implemented based on 4 families:
- Machine learning agents: SVM, Linear Regression, ...
- Deep Learning
- Embedding
- Dot product

The relevent agents also implement an online learning.

## Results

It seems that the Deep Learning and Linear Regression agents don't perform well on the environment. But the SVM, Gradient boosting tree, embedding (when the environment is not evolutive) and the dot agents perform really well. Because the reward is partly based on a dot product, the embedding and dot agents results are not surprising, but the SVM and gradient boosting results were better than expected.

## Link to the course materials

This project has been developed as part of a class taught by Thibault Allart. The course material and the links to the other projets are available [here](https://github.com/thibaultallart/IA316-2020). 
