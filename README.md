# Applied Machine Learning
**What is Machine Learning ?** ML is an area of AI research providing powerful tools that enable machines (computers) to find 
**models** describing data and the correlations between them, to
* **predict** future outcomes or developments given previous data (weather, stock market)
* **classify** unknown input given known classified data (scene contains pedestrian or not) 
* **identify** structures in unseen, unlabeled data (grouping people according to different attributes)
* **decide** upon next steps, or actions to take, to maximise reward (new measurement, robot action)

## Contents
### Basics
* **Preliminaries**: Math and Python (lecture 2, lab 1)
* **Fundamental ML techniques**:  Concept Learning, Clustering, K-Nearest Neighbour (KNN), Self Organising Maps (SOM), <br>
 Decision Trees, Feedforward Networks, Loss, Regularization, … (lectures 3-5, labs 2+3)
### Specific approaches
* **Deep learning techniques**: Convolutional NNs, Recurrent NNs, LSTMs and GRUs, Autoencorders (lectures 6-9, lab 4)
* **Distributed techniques**: Spark / GPU (lecture 10, lab 5)
* **Probabilistic methods, Bayesian learning**: MAP-Learning, NBC, GMMs (lectures 11+12, lab 6)
* **Reinforcement Learning**: Introduction, TD Learning, Q-Learning, Actor Critic, Applications in Robotics (lectures 13+14, lab 7)

## Assignments
### Assignment 1
 In this lab, we will only review the Python syntax, linear algebra, and important numpy features.
### Assignment 2
The goal of this lab assignment is to get acquainted with **WEKA**, *Waikato Environment for Knowledge Analysis*, and to learn how to load data sets into the WEKA tool. Then, we will experiment with some **clustering** and **classification** algorithms. <br>
In addition, in the second part of the assignment, we will explore some programming concepts in Python, SciKitLearn, and partially Numpy to implement a **decision tree** classifier and compare it with the provided one in SciKitLearn. Then, we will get acquainted with the simplified version of the MNIST dataset provided in SciKitLearn, and explore the effect of (some) data preprocessing on the learning process.
### Assignment 3
The objectives of this assignment are to Write a program to recognize flowers on images, learn how to manage an image data set, apply convolutional networks to images, know what Python generators are, and understand class activation. You will have to experiment different architectures and compare the results you obtained.
### Assignment 4
The objectives of this assignment are to write a program to recognize named entities in text, learn how to manage a text data set, apply recurrent neural networks to text, and know what word embeddings are. You will have to experiment different architectures, namely RNN and LSTM, and compare the results you obtained.
### Assignment 5
In this assignment, we Learn how to read, transform and process text data with **Pyspark**, preprocess and create a suitable dataset for **clustering**, use **KMeans** from **sklearn** and cluster 10 000 words to 200 clusters, and write a function which displays words nearby. More specifically, you will first solve a few exercises on *Spark* to learn how to write basic commands. You will then apply *Spark* to extract the 10,000 most frequent words in the English Wikipedia. As this corpus is very large, you will use 1% of it in the lab, the full Wikipedia is available. You will finally cluster these words into 100 groups using their GloVe100 representation. As clustering program, you will use *KMeans* from *sklearn*.
### Assignment 6
In and after this lab session you will train a **Gaussian NBC** with the EM algorithm, compare the results you get to those of the **k-Means clustering** provided in SciKitLearn, and discuss the classifiers from this lab session and those from the previous session (supervised learning of NBCs) in a brief report.

### Assignment 7
In this lab you are going to implement a **deep reinforcement learning** agent. Unlike in **supervised learning** the algorithm does not learn from examples but rather from interacting with the problem, i.e. trial and error.

## Sources
* source 0: Richard Johansson “Scientific Computing with Python” <br>
 https://github.com/jrjohansson/scientific-python-lectures

* source 1: Aurélien Géron https://github.com/ageron/handson-ml
* source 2: François Chollet https://github.com/fchollet/deep-learning-with-python-notebooks

## Literature
* Kevin P. Murphy: Machine Learning, "A Probabilistic Perspective". MIT Press, 2012, ISBN: 9780262018029.
* Ian Goodfellow, Yoshua Bengio, Aaron Courville: "Deep Learning". MIT Press, 2016, ISBN: 
9780262035613.
* Aurélien Géron: "Hands-On Machine Learning with Scikit-Learn and TensorFlow, Concepts, Tools, and Techniques to Build Intelligent Systems". O'Reilly Media, 2017, ISBN: 9781491962299.

* François Chollet: "Deep Learning with Python". Manning, 2018, ISBN: 9781617294433.
* Tom Mitchell: "Machine Learning". McGraw Hill, 1997, ISBN: 0070428077.
* David L. Poole, Alan K. Mackworth: "Artificial Intelligence - Foundations of Computational Agents (2e)". Cambridge University Press, 2017, ISBN:9781107195394.

* Stuart Russel, Peter Norvig: Artificial Intelligence - A Modern Approach (3e). Pearson, 2010, ISBN:10: 0132071487
