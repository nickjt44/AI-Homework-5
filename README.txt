**************** README for Assignment 5 part 2 ****************

This README describes the implementation of the decision tree classifier for Assignment 5, part 2.

The code was all implemented in Python, using the library scikit-learn, which also requires the
numpy and scipy libraries.  I used the WinPython distribution, because I had trouble installing
scikit-learn for my existing version of Python.

My code consists of a single file, aihw5.py.   The file includes a number of functions, all of which
are described by comments, and the code required to run the functions.

The relevant functions are treeLearn, which is the learner for a unigram model, treeLearnBigram,
for the bigram model, and treeLearnTrigram, for the trigram model. Also loadAllData() and loadListData().

loadAllData() is called before anything else, it takes data from all files, and splits it into a
training and testing set at a 70/30 ratio, and stores it in two separate files
named trainingdata.txt and testdata.txt. This function should only be called once, and then commented out. Or not
at all, since I will include my training and test files with my submission.

loadListData() is called once with trainingdata.txt as an argument, and once with testdata.txt as an argument. The
function converts the file data into a more manageable form for scikit to deal with, and outputs the formatted data.

treeLearn then takes the formatted training data and formatted test data as arguments, builds the classifier (with a
unigram model), and predicts the values for the test data, and prints results.

The bigram and trigram functions do exactly the same, just using bigram and trigram models instead. Three functions weren't
necessary, but I believe separate functions for each process made it more clear what one is supposed to do.

Simply run the code in Python's IDE, all files are in the same folder.

The bigram and trigram functions take a few seconds (up to around 20 each) to run.

