//
//  main.cpp
//  Neural Logic Circuits - NLC - v1
//
//  Created by Hamit Taner Ünal on 30.04.2021.
//  Copyright © 2021 Hamit Taner Ünal & Prof.Fatih Başçiftçi. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <sstream>
#include <array>
#include <sys/time.h>
#include <stdlib.h>

using namespace std;

inline const char * const BoolToString(bool b);
void write_csv(std::string filename, std::string colname, std::vector<int> vals);
int random(int from, int to);
void swap (bool *a, bool *b);
void randomShuffle (bool arr[], int n);
void swapInt (int *a, int *b);
void randomShuffleInt (int arr[], int n);
void printArray (bool arr[], int n);
bool getAndMask(bool maskArray[], bool outputArray[], int n);
bool getOrMask(bool maskArray[], bool outputArray[], int n);
bool getNAndMask(bool maskArray[], bool outputArray[], int n);
bool getNOrMask(bool maskArray[], bool outputArray[], int n);
bool getXOrMask(bool maskArray[], bool outputArray[], int n);
bool getXNOrMask(bool maskArray[], bool outputArray[], int n);
int findMax(float arr[], int n, int gateCount[], int popIndex[]);
int seed;
int fold;

int main(int argc, const char * argv[]) {

    //Randomize device (get best random numbers)
    struct timeval time;
    gettimeofday(&time,NULL);
    //srand(time.tv_usec);

    //Randomize with a number
    seed = 0;
    fold = 0;
    srand(seed);

    //Read CSV File
    std::ifstream myFile("pima2.csv");
    std::string line, colName;
    int val;
    int colNumber=0;
    int numberOfTotalRows = 0;

    //Get column names, number of columns, number of rows
    if(myFile.good())
    {
        // Extract the first line in the file
        std::getline(myFile, line);

        // Create a stringstream from line
        std::stringstream ss(line);

        printf("Column Names\n");
        // Extract each column name
        while(std::getline(ss, colName, ',')){
            colNumber++;

            // Print colnames
            printf("%s", colName.c_str());
            printf(", ");
        }
        printf("\n");

        //Get row count:
        std::string unused;
        while ( std::getline(myFile, unused) )
            ++numberOfTotalRows;
    }
    myFile.close();
    int numberOfInputBits = colNumber - 1;//except column 0

    //Now, let's print the results
    printf("Number of cols:");printf("%d", numberOfInputBits);printf("\n");
    printf("Number of rows:");printf("%d", numberOfTotalRows);printf("\n");

    //Define dataset variables (arrays)
    bool X_raw[numberOfTotalRows][numberOfInputBits];
    bool y_raw[numberOfTotalRows][1];

    bool X_raw_shuffled[numberOfTotalRows][numberOfInputBits];
    bool y_raw_shuffled[numberOfTotalRows][1];

    //Read the file again and fill dataset variables with data
    std::ifstream myFileAgain("pima2.csv");
    std::getline(myFileAgain, line);

    int rowID=0;
    // Read data, line by line
    while(std::getline(myFileAgain, line))
    {
        // Create a string stream of the current line
        std::stringstream ss(line);
        // Keep track of the current column index
        int colIdx = 0;
        // Extract each integer
        while(ss >> val){
            if (colIdx != numberOfInputBits) X_raw[rowID][colIdx]=(bool)val;//Fill input variables
            else y_raw[rowID][0] = (bool)val;//Fill output class
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
            // Increment the column index
            colIdx++;
        }
        rowID++;
    }
    // Close file
    myFileAgain.close();
    printf("\n");

    //Partition data to k-folds
    //**************************
    //Shuffle Raw Data first
    int shuffledArrayIndex[numberOfTotalRows];
    for (int i=0; i < numberOfTotalRows; i++) shuffledArrayIndex[i]=i;
    randomShuffleInt(shuffledArrayIndex, numberOfTotalRows);

    //Fill shuffled data
    for (int i=0; i < numberOfTotalRows; i++) {
        for (int j=0; j < numberOfInputBits; j++) {
            X_raw_shuffled[i][j] = X_raw[shuffledArrayIndex[i]][j];
        }
        y_raw_shuffled[i][0] = y_raw[shuffledArrayIndex[i]][0];
    }

    //Create Stratified k-fold segments
    int numberOfK = 10;
    int numberOfSamplesForEachK = (int)(numberOfTotalRows / numberOfK);
    printf("Number of k=%d\n",numberOfK);
    printf("K Samples:%d\n", numberOfSamplesForEachK);

    //Let's calculate the number and proportion of 1s and 0s in the dataset
    int numberOf1s=0;
    int numberOf0s=0;

    for (int i=0; i < numberOfTotalRows; i++)
    {
        if (y_raw[i][0]==1) numberOf1s++;
        if (y_raw[i][0]==0) numberOf0s++;
    }

    printf("Number of 1s:%d\n",numberOf1s);
    printf("Number of 0s:%d\n",numberOf0s);

    //Class distribution on each stratified k-fold
    int oneForEachFold = (int)(numberOf1s / numberOfK);
    int zeroForEachFold = (int)(numberOf0s / numberOfK);

    printf("1 for each fold:%d\n",oneForEachFold);
    printf("0 for each fold:%d\n",zeroForEachFold);

    //Define k-fold arrays
    bool **kTrain_X[numberOfK];
    bool **kTest_X[numberOfK];

    bool **kTrain_y[numberOfK];
    bool **kTest_y[numberOfK];

    int *kTestIndexes[numberOfK];

    //Initialize k-fold arrays
    for (int i=0; i < numberOfK; i++)
    {
        //Initialize number of samples for each fold
        kTest_X[i] = new bool *[numberOfSamplesForEachK];
        kTrain_X[i] = new bool *[numberOfTotalRows - numberOfSamplesForEachK];

        kTest_y[i] = new bool *[numberOfSamplesForEachK];
        kTrain_y[i] = new bool *[numberOfTotalRows - numberOfSamplesForEachK];

        //We will store test indexes later
        kTestIndexes[i] = new int [numberOfSamplesForEachK];

        //Initialize test cols
        for (int j=0; j < numberOfSamplesForEachK; j++)
        {
            kTest_X[i][j] = new bool [numberOfInputBits];
            kTest_y[i][j] = new bool [1];
        }

        //Initialize train cols
        for (int j=0; j < numberOfTotalRows - numberOfSamplesForEachK; j++)
        {
            kTrain_X[i][j] = new bool [numberOfInputBits];
            kTrain_y[i][j] = new bool [1];
        }
    }

    //Define loop parameters
    int currentKFold1s=0;
    int currentKFold0s=0;
    int curr1s=0;
    int curr0s=0;
    int currentSample[numberOfK];

    for (int j=0; j < numberOfK; j++)
        currentSample[j]=0;

    //Fill number of k-fold arrays and record indexes
    for (int i=0; i < numberOfTotalRows; i++)
    {
        if (y_raw_shuffled[i][0]==1 && currentKFold1s < numberOfK) {
            for (int j = 0; j < numberOfInputBits; j++) {
                kTest_X[currentKFold1s][currentSample[currentKFold1s]][j] = X_raw_shuffled[i][j];
            }
            kTest_y[currentKFold1s][currentSample[currentKFold1s]][0] = y_raw_shuffled[i][0];
            kTestIndexes[currentKFold1s][currentSample[currentKFold1s]] = i;
            curr1s++;
            currentSample[currentKFold1s]++;
        }

        if (y_raw_shuffled[i][0]==0 && currentKFold0s < numberOfK) {
            for (int j = 0; j < numberOfInputBits; j++) {
                kTest_X[currentKFold0s][currentSample[currentKFold0s]][j] = X_raw_shuffled[i][j];
            }
            kTest_y[currentKFold0s][currentSample[currentKFold0s]][0] = y_raw_shuffled[i][0];
            kTestIndexes[currentKFold0s][currentSample[currentKFold0s]] = i;
            curr0s++;
            currentSample[currentKFold0s]++;
        }

        if (curr1s==oneForEachFold)
        {
            currentKFold1s++;
            curr1s=0;
        }

        if (curr0s==zeroForEachFold)
        {
            currentKFold0s++;
            curr0s=0;
        }

    }

    //Now, let's fill train sets (with remaining data)
    for (int i=0; i < numberOfK; i++)
    {
        int currentTrainRow=0;
        for (int j=0; j < numberOfTotalRows; j++)
        {
            bool isFoundInTest=false;
            for (int m=0; m < numberOfSamplesForEachK; m++)
            {
                if (j==kTestIndexes[i][m]) {
                    isFoundInTest= true;//Skip row
                }
            }
            if (!isFoundInTest)
            {
                for (int n=0; n < numberOfInputBits; n++)
                {
                    kTrain_X[i][currentTrainRow][n] = X_raw_shuffled[j][n];
                }
                kTrain_y[i][currentTrainRow][0] = y_raw_shuffled[j][0];
                currentTrainRow++;
            }
        }
    }


    //Let's Define GA Parameters
    //Determine population size manually
    int populationSize = 300;
    //Number of gates for each population
    int gateCount[populationSize];
    //Determine iteration count manually
    int iterationCount = 6000;
    //Define initial max number of gates. This number will be increased gradually over iterations
    float initialMaxGates = 1.0f;
    //Number of max connections for each gate. Determined randomly (minimum=2)
    int connectMax = 18;
    //Are we going to use XOR gates (XOR-XNOR)?
    bool useXOR= true;
    int maxGateType;
    if (useXOR) maxGateType=5;else maxGateType=3;

    //Define column strings
    //popBits are connection masks for each gate. 1 denotes a connection and 0 denotes no connection
    bool **popBits[populationSize];
    //popGateType are gate types (AND, OR, XOR etc.)
    int *popGateType[populationSize];

    //Population accuracies
    float popAcc[populationSize];

    //Tournament Selection Parameter
    float tournamentRate = 0.05f;
    int tournamentCount = (int)((float)populationSize*tournamentRate);

    //Crossover Parameters
    float crossoverRate = 0.75f;
    float gateTypeCrossoverRate = 1.0f;//This is a new parameter checking if gateType will be crossed during crossover (together with the bits)

    //Mutation Parameters
    float mutationRate = 0.25f;
    float gateMutationRate = 1.0f;//A new parameter to mutate gateType

    //Elitism parameter
    float elitismRate = 0.03f;
    int popCountElite = (int)((float)populationSize*elitismRate);

    //Augmentation parameters
    float augmentationRate = 1.0005f;//How the number of gates will increase
    int maxGatesWithAugmentation;//TBD
    float augmentationPopRate = 0.05f;//What portion of population will be replaced with new, augmented networks
    int popCountToBeAugmented = (int)(populationSize * augmentationPopRate);

    //Evaluation Parameters
    bool useKFold= true;
    int currentKFold=fold;

    //Let's start GA!
    //Create Initial Population
    for (int popIndex=0; popIndex < populationSize; popIndex++)
    {
        //Determine random number of gates
        gateCount[popIndex] = random(1, (int)initialMaxGates);
        popBits[popIndex] = new bool*[gateCount[popIndex]];
        popGateType[popIndex] = new int[gateCount[popIndex]];

        //Create -for loop- for each gate
        //Loop starts with gate0 (after input columns)
        //Let's generate gate content
        for (int gateIndex=0; gateIndex < gateCount[popIndex]; gateIndex++)
        {
            //Determine gate type
            // 0:AND
            // 1:OR
            // 2:NAND
            // 3:NOR
            // 4:XOR
            // 5:XNOR
            popGateType[popIndex][gateIndex]= random(0, maxGateType);
            //Let's start randomly filling gate masks
                //Define sub array
                //The length of array is number of inputs+gate index (gateindex+numberOfInputBits)
                popBits[popIndex][gateIndex]=new bool [gateIndex + numberOfInputBits];
                //Determine connections count randomly
                int connectionsCount = random(2, connectMax);
            //Fill connection count times True (1)
            if ((gateIndex+numberOfInputBits)<=connectionsCount)
            {
                for (int i=0;i<(numberOfInputBits+gateIndex);i++) popBits[popIndex][gateIndex][i]=true;
            }
                //Fill the rest with false
            else {
                for (int i=0;i<connectionsCount;i++) popBits[popIndex][gateIndex][i]=true;
                for (int i=connectionsCount; i < (gateIndex + numberOfInputBits); i++) popBits[popIndex][gateIndex][i]=false;
            }
                //Now shuffle 1s with 0s - there you have a random mask! We call this Neuroplasticity...
                //Our brain is doing it many times a day :)
                randomShuffle(popBits[popIndex][gateIndex], gateIndex + numberOfInputBits);
                //printArray(popBits[popIndex][gateIndex], gateIndex + numberOfInputBits);

        }//End of gateIndex
    }//End of popIndex



    //First, we will define new population variables
    bool **popBitsAfterMutation[populationSize];
    int *popGateTypeAfterMutation[populationSize];
    int gateCountAfterMutation[populationSize];

    float testAccOnIterations;

    float bestTestAcc=0;
    int bestTestAccIndex;
    int bestTestAccIteration;
    bool bestTestAccChanged;

    int bestTP = 0;
    int bestFP = 0;
    int bestTN = 0;
    int bestFN = 0;

    float bestRecall=0;
    float bestPrecision=0;
    float bestF_Score=0;
    float bestSpecificity=0;
    float bestFalsePositiveRate;

    float bestBalancedAccuracy;

    float trainAccOnEachIteration[iterationCount];
    float testAccOnEachIteration[iterationCount];


    float bestTrainAcc = 0.0f;
    int bestTrainAccIndex;
    int bestTrainAccIteration;
    bool bestTrainAccChanged;

    //Record best train acc
    float bestTrainAccKFold[numberOfK];
    int maxIterationForKFold=300;

    //Let's start GA loop
    for (int iterationIndex=0;iterationIndex<iterationCount;iterationIndex++)
    {
        //Update maxGates for augmentation
        initialMaxGates *= augmentationRate;//5.2f;//*= (float)augmentationRate;


        //SITREP for Iteration
        printf("Iteration: %d,\n",iterationIndex);
        printf("Max Gates Float:%.2f\n",initialMaxGates);
        printf("Max Gates:%d\n",(int)initialMaxGates);

        //Re-initialize loop (transfer values from the last iteration)
        if (iterationIndex!=0)
        {
            for (int popIndex=0;popIndex<populationSize;popIndex++) {
                gateCount[popIndex] = gateCountAfterMutation[popIndex];
                popBits[popIndex] = new bool *[gateCount[popIndex]];
                popGateType[popIndex] = new int[gateCount[popIndex]];

                for (int gateIndex=0;gateIndex<gateCount[popIndex];gateIndex++)
                {
                    popGateType[popIndex][gateIndex] = popGateTypeAfterMutation[popIndex][gateIndex];
                    popBits[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];
                    for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                        popBits[popIndex][gateIndex][i] = popBitsAfterMutation[popIndex][gateIndex][i];
                }
            }
        }

        //Calculate accuracies of each population (train)
        if (useKFold)
        {
            //Use kFold
            float bestTrainAccInPop=0;
            for (int popIndex = 0; popIndex < populationSize; popIndex++) {
                int numberOfCorrect = 0;
                //Define bit output
                bool bitOutput[numberOfInputBits + gateCount[popIndex]];
                //Start with iterating through input rows
                for (int rowIndex = 0; rowIndex < numberOfTotalRows - numberOfSamplesForEachK; rowIndex++) {
                    //First fill bitOutput with inputs
                    for (int colIndex = 0; colIndex < numberOfInputBits; colIndex++) {
                        bitOutput[colIndex] = kTrain_X[currentKFold][rowIndex][colIndex];
                    }
                    //Then, iterate through gates and calculate gate outputs
                    for (int gateIndex = 0; gateIndex < gateCount[popIndex]; gateIndex++) {
                        switch (popGateType[popIndex][gateIndex]) {
                            //AND Gate
                            case 0:
                                bitOutput[numberOfInputBits + gateIndex] = getAndMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                      (gateIndex + numberOfInputBits));
                                break;
                            //OR Gate
                            case 1:
                                bitOutput[numberOfInputBits + gateIndex] = getOrMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                     (gateIndex + numberOfInputBits));
                                break;
                            //NAND Gate
                            case 2:
                                bitOutput[numberOfInputBits + gateIndex] = getNAndMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                       (gateIndex + numberOfInputBits));
                                break;
                            //NOR Gate
                            case 3:
                                bitOutput[numberOfInputBits + gateIndex] = getNOrMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                      (gateIndex + numberOfInputBits));
                                break;
                            //XOR Gate
                            case 4:
                                bitOutput[numberOfInputBits + gateIndex] = getXOrMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                      (gateIndex + numberOfInputBits));
                                break;
                            //XNOR Gate
                            case 5:
                                bitOutput[numberOfInputBits + gateIndex] = getXNOrMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                       (gateIndex + numberOfInputBits));
                                break;
                        }//end of gate switch
                    }//end loop for gate index

                    //Now, we can check if the last gate (output) is equal to expected output
                    if (bitOutput[numberOfInputBits + gateCount[popIndex] - 1] == kTrain_y[currentKFold][rowIndex][0]) numberOfCorrect++;
                }//end loop for Rows
                popAcc[popIndex] = (float) numberOfCorrect / (float) (numberOfTotalRows - numberOfSamplesForEachK);
                if (popAcc[popIndex]>bestTrainAccInPop)
                    bestTrainAccInPop = popAcc[popIndex];
            }//end loop for popIndex (Calculate accuracies)
            trainAccOnEachIteration[iterationIndex] = bestTrainAccInPop;
        }

        //Calculate test accuracy and report
        if (useKFold)
        {
            float bestTestAccInPop=0;
            for (int popIndex = 0; popIndex < populationSize; popIndex++) {
                int numberOfCorrect = 0;
                int TP = 0;
                int FP = 0;
                int TN = 0;
                int FN = 0;
                //Define bit output
                bool bitOutput[numberOfInputBits + gateCount[popIndex]];
                //Start with iterating through input rows
                for (int rowIndex = 0; rowIndex < numberOfSamplesForEachK; rowIndex++) {
                    for (int colIndex = 0; colIndex < numberOfInputBits; colIndex++) {
                        bitOutput[colIndex] = kTest_X[currentKFold][rowIndex][colIndex];
                    }
                    //Then, iterate through gates and calculate gate outputs
                    for (int gateIndex = 0; gateIndex < gateCount[popIndex]; gateIndex++) {
                        switch (popGateType[popIndex][gateIndex]) {
                            //AND Gate
                            case 0:
                                bitOutput[numberOfInputBits + gateIndex] = getAndMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                      (gateIndex + numberOfInputBits));
                                break;
                            //OR Gate
                            case 1:
                                bitOutput[numberOfInputBits + gateIndex] = getOrMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                     (gateIndex + numberOfInputBits));
                                break;
                            //NAND Gate
                            case 2:
                                bitOutput[numberOfInputBits + gateIndex] = getNAndMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                       (gateIndex + numberOfInputBits));
                                break;
                            //NOR Gate
                            case 3:
                                bitOutput[numberOfInputBits + gateIndex] = getNOrMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                      (gateIndex + numberOfInputBits));
                                break;
                            //XOR Gate
                            case 4:
                                bitOutput[numberOfInputBits + gateIndex] = getXOrMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                      (gateIndex + numberOfInputBits));
                                break;
                            //XNOR Gate
                            case 5:
                                bitOutput[numberOfInputBits + gateIndex] = getXNOrMask(popBits[popIndex][gateIndex], bitOutput,
                                                                                       (gateIndex + numberOfInputBits));
                                break;
                        }//end of gate switch
                    }//end loop for gate index

                    //Now, we can check if the last gate (output) is equal to expected output
                    if (bitOutput[numberOfInputBits + gateCount[popIndex] - 1] == kTest_y[currentKFold][rowIndex][0]) numberOfCorrect++;
                    //Calculate confusion matrix variables
                    if (kTest_y[currentKFold][rowIndex][0]==1) //Positive class
                    {
                        //TP:The model predicted the positive class correctly, to be a positive class.
                        if (bitOutput[numberOfInputBits + gateCount[popIndex] - 1] == kTest_y[currentKFold][rowIndex][0]) TP++;
                        //FN:The model predicted the positive class incorrectly, to be the negative class. (Type-2 Error)
                        else FN++;
                    }
                    else //Negative class
                    {
                        //TN:The model predicted the negative class correctly, to be the negative class.
                        if (bitOutput[numberOfInputBits + gateCount[popIndex] - 1] == kTest_y[currentKFold][rowIndex][0]) TN++;
                        //FP:The model predicted the negative class incorrectly, to be a positive class. (Type-1 Error)
                        else FP++;
                    }
                }//end loop for Rows
                float testAcc = (float) numberOfCorrect / (float) numberOfSamplesForEachK;
                float recall = (float) TP / ((float) (TP+FN));
                float precision = (float) TP / ((float) (TP+FP));
                float f1_score = (2.0f * recall * precision) / (recall + precision);
                float specificity = float (TN) / ((float) (TN+FP));
                float balancedAccuracy = ((float)(recall + specificity)) / 2.0f;
                float falsePositiveRate = (float) FP / ((float)(FP+TN));

                //Record best test acc in population
                if (testAcc>bestTestAccInPop)
                    bestTestAccInPop=testAcc;

                //Record best test acc overall
                if (testAcc > bestTestAcc) {
                    bestTestAcc = testAcc;
                    bestTestAccIndex = popIndex;
                    bestTestAccIteration = iterationIndex;
                    bestTestAccChanged = true;



                    bestTP = TP;
                    bestTN = TN;
                    bestFP = FP;
                    bestFN = FN;

                    bestRecall = recall;
                    bestPrecision = precision;
                    bestF_Score = f1_score;
                    bestSpecificity = specificity;
                    bestBalancedAccuracy = balancedAccuracy;
                    bestFalsePositiveRate = falsePositiveRate;

                    printf("\n*************** Printing Gates for the new best on Test Accuracy **********\n");
                    printf("Iteration: %d\n",bestTestAccIteration);
                    printf("Population member: %d\n",bestTestAccIndex);
                    printf("Train accuracy: %f\n",popAcc[bestTestAccIndex]);
                    printf("Test accuracy: %f\n",bestTestAcc);
                    printf("Best TP: %d\n",bestTP);
                    printf("Best TN: %d\n",bestTN);
                    printf("Best FP: %d\n",bestFP);
                    printf("Best FN: %d\n",bestFN);
                    printf("Recall (Sensitivity-TPR): %.8f\n",bestRecall);
                    printf("Precision - PPV: %.8f\n",bestPrecision);
                    printf("Specificity - TNR: %.8f\n",bestSpecificity);
                    printf("False Positive Rate (FPR) (1-Specificity): %.8f\n",bestFalsePositiveRate);
                    printf("F1-Score: %.8f\n",bestF_Score);
                    printf("Balanced Accuracy: %.8f\n",bestBalancedAccuracy);
                    printf("Gate count for best pop: %d\n",gateCount[bestTestAccIndex]);
                    printf("--------------------------------------------------\n");
                    for (int gIndex=0;gIndex<gateCount[bestTestAccIndex];gIndex++)
                    {
                        printf("Gate: %d",gIndex);
                        if (gIndex==gateCount[bestTestAccIndex]-1) printf(" (Output gate)");
                        printf("\n");
                        printf("Gate type: %d\n",popGateType[bestTestAccIndex][gIndex]);
                        printf("Gate connections:\n");
                        printArray(popBits[bestTestAccIndex][gIndex], gIndex + numberOfInputBits);
                    }
                    printf("*************** End of Printing Gates **********\n");
                }

            }//end loop for test accuracy
            printf("..Best Test Accuracy: %f at Iteration %d, Pop %d, with Train acc:%f\n", bestTestAcc, bestTestAccIteration,
                   bestTestAccIndex,popAcc[bestTestAccIndex]);
            testAccOnEachIteration[iterationIndex] = bestTestAccInPop;


        }

        //Now we sort the population based on training accuracies
        float tempValue;
        int tempIndex;

        //Define popIndex Original and initialize
        int sortedPopIndex[populationSize];
        for (int i=0;i<populationSize;i++)
            sortedPopIndex[i]=i;//Not yet sorted. We just initialized in ascending order.

        //Do a quick bubble sort and record indexes
        for(int i=0;i<populationSize-1;i++)
        {
            for(int j=0;j<populationSize-i-1;j++)
            {
                if(popAcc[j]>popAcc[j+1])
                {
                    tempValue=popAcc[j+1];
                    tempIndex = sortedPopIndex[j + 1];
                    popAcc[j+1]=popAcc[j];
                    sortedPopIndex[j + 1]=sortedPopIndex[j];
                    popAcc[j]=tempValue;
                    sortedPopIndex[j]=tempIndex;
                }
            }
        }

        //Now move everything to new (sorted) population
        //Let's start with defining new variables for sorted population
        bool **popBitsSorted[populationSize];
        int *popGateTypeSorted[populationSize];
        int gateCountSorted[populationSize];

        //Ok, now population accuracies are already sorted and we have the indexes
        //Now we will form the new population in three steps
        //First we will start with newly created individuals with augmented gates
        //They will fill the slots of the worst population members, coming in the first place

        for (int popIndex=0;popIndex<popCountToBeAugmented;popIndex++)
        {

            gateCountSorted[popIndex] = random(1, (int)initialMaxGates);
            popBitsSorted[popIndex] = new bool*[gateCountSorted[popIndex]];
            popGateTypeSorted[popIndex] = new int[gateCountSorted[popIndex]];

            //Let's generate gate content for newly created individuals
            for (int gateIndex=0; gateIndex < gateCountSorted[popIndex]; gateIndex++)
            {
                //Determine gate type
                // 0:AND
                // 1:OR
                // 2:NAND
                // 3:NOR
                // 4:XOR
                // 5:XNOR
                popGateTypeSorted[popIndex][gateIndex]= random(0, maxGateType);

                //Let's start randomly filling gate masks
                    //Define sub array
                    //The length of array is number of inputs+gate index (gateindex+numberOfInputBits)
                    popBitsSorted[popIndex][gateIndex]=new bool [gateIndex + numberOfInputBits];
                    //Generate a random number for connections count
                    int connectionsCount = random(2, connectMax);
                //Fill connection times true
                if ((gateIndex+numberOfInputBits)<=connectionsCount)
                {
                    for (int i=0;i<(numberOfInputBits+gateIndex);i++) popBitsSorted[popIndex][gateIndex][i]=true;
                }
                    //Fill the rest with false
                else {
                    for (int i=0;i<connectionsCount;i++) popBitsSorted[popIndex][gateIndex][i]=true;
                    for (int i=connectionsCount; i < (gateIndex + numberOfInputBits); i++) popBitsSorted[popIndex][gateIndex][i]=false;
                }
                    //Now shuffle'em! There you have a randomized neural connections.
                    randomShuffle(popBitsSorted[popIndex][gateIndex], gateIndex + numberOfInputBits);//Shuffle contents
            }//End of gateIndex
        }

        //Now, calculate training accuracies of newly introduced population members (wish us luck!)
        if (useKFold)
        {
            for (int popIndex = 0; popIndex < popCountToBeAugmented; popIndex++) {

                int numberOfCorrect = 0;

                //Define bit output for each gate (including inputs)
                bool bitOutput[numberOfInputBits + gateCountSorted[popIndex]];
                //Start with iterating through input rows
                for (int rowIndex = 0; rowIndex < numberOfTotalRows - numberOfSamplesForEachK; rowIndex++) {
                    //First fill bitOutput with inputs
                    for (int colIndex = 0; colIndex < numberOfInputBits; colIndex++) {
                        bitOutput[colIndex] = kTrain_X[currentKFold][rowIndex][colIndex];
                    }

                    //Then, iterate through gates and calculate gate outputs
                    for (int gateIndex = 0; gateIndex < gateCountSorted[popIndex]; gateIndex++) {
                        switch (popGateType[popIndex][gateIndex]) {
                            //AND Gate
                            case 0:
                                bitOutput[numberOfInputBits + gateIndex] = getAndMask(popBitsSorted[popIndex][gateIndex],
                                                                                      bitOutput, (gateIndex + numberOfInputBits));
                                break;
                                //OR Gate
                            case 1:
                                bitOutput[numberOfInputBits + gateIndex] = getOrMask(popBitsSorted[popIndex][gateIndex],
                                                                                     bitOutput, (gateIndex + numberOfInputBits));
                                break;
                                //NAND Gate
                            case 2:
                                bitOutput[numberOfInputBits + gateIndex] = getNAndMask(popBitsSorted[popIndex][gateIndex],
                                                                                       bitOutput, (gateIndex + numberOfInputBits));
                                break;
                                break;
                                //NOR Gate
                            case 3:
                                bitOutput[numberOfInputBits + gateIndex] = getNOrMask(popBitsSorted[popIndex][gateIndex],
                                                                                      bitOutput, (gateIndex + numberOfInputBits));
                                break;
                                //XOR Gate
                            case 4:
                                bitOutput[numberOfInputBits + gateIndex] = getXOrMask(popBitsSorted[popIndex][gateIndex],
                                                                                      bitOutput, (gateIndex + numberOfInputBits));
                                break;
                                //XNOR Gate
                            case 5:
                                bitOutput[numberOfInputBits + gateIndex] = getXNOrMask(popBitsSorted[popIndex][gateIndex],
                                                                                       bitOutput, (gateIndex + numberOfInputBits));
                                break;
                        }//end of gate switch
                    }//end loop for gate index
                    //Now, we can check if the last gate (output) is equal to expected output
                    if (bitOutput[numberOfInputBits + gateCount[popIndex] - 1] == kTrain_y[currentKFold][rowIndex][0]) numberOfCorrect++;
                }//end loop for Rows
                //And, here is the result..
                popAcc[popIndex] = (float) numberOfCorrect / (float) (numberOfTotalRows - numberOfSamplesForEachK);
            }//end loop for popIndex (Calculate accuracies of newly introduced population members)
        }

        //Now, move existing population contents to sorted population
        for (int popIndex=popCountToBeAugmented;popIndex<populationSize;popIndex++)
        {
            gateCountSorted[popIndex] = gateCount[sortedPopIndex[popIndex]];
            popBitsSorted[popIndex] = new bool*[gateCountSorted[popIndex]];
            popGateTypeSorted[popIndex] = new int[gateCountSorted[popIndex]];

            //Iterate through each gate. Moving is always hard!
            for (int gateIndex=0;gateIndex<gateCountSorted[popIndex];gateIndex++)
            {
                //Define new array for popBits
                popBitsSorted[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];
                //Get gateType from old index
                popGateTypeSorted[popIndex][gateIndex] = popGateType[sortedPopIndex[popIndex]][gateIndex];
                //Now let's move popBits to popBitsSorted
                for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                    popBitsSorted[popIndex][gateIndex][i] = popBits[sortedPopIndex[popIndex]][gateIndex][i];
            }
        }

        //We can start Tournament Selection
        //First, we will create new population variables
        bool **popBitsAfterTournament[populationSize];
        int *popGateTypeAfterTournament[populationSize];
        int gateCountAfterTournament[populationSize];

        //Iterate populationSize-popCountElite times (Careful: elites can be selected as winners but do not override the elite!)
        for (int popIndex=0;popIndex<populationSize-popCountElite;popIndex++)
        {
            //Define temp array for tournament candidates
            int tempTourIndex[tournamentCount];
            float tempTourAcc[tournamentCount];

            //Pick random members and add to array
            for (int i=0;i<tournamentCount;i++)
            {
                int randomIndex = random(0,populationSize-1);//Elites can also be selected in the tournament
                tempTourIndex[i] = randomIndex;
                tempTourAcc[i] = popAcc[randomIndex];
            }
            //Find max acc among selected tournament candidates
            //Here, a good thing is, if acc of two pop members is the same, then findMax function selects the one with fewer gates (SPARSITY!)
            //And the Oscar goes to....:
            int tournamentWinner = tempTourIndex[findMax(tempTourAcc, tournamentCount, gateCountSorted, tempTourIndex)];

            //Now, fill new population with the winner
            //Initialize variables
            gateCountAfterTournament[popIndex] = gateCountSorted[tournamentWinner];
            popBitsAfterTournament[popIndex] = new bool*[gateCountSorted[tournamentWinner]];
            popGateTypeAfterTournament[popIndex] = new int[gateCountSorted[tournamentWinner]];

            //Move winner to new pop
            //Iterate through each gate..
            for (int gateIndex=0;gateIndex<gateCountAfterTournament[popIndex];gateIndex++)
            {
                //Define new array for popBits
                popBitsAfterTournament[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];
                //Get gateType from old index
                popGateTypeAfterTournament[popIndex][gateIndex] = popGateTypeSorted[tournamentWinner][gateIndex];
                //Now let's move previous popBits to popBitsAfterTournament
                for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                    popBitsAfterTournament[popIndex][gateIndex][i] = popBitsSorted[tournamentWinner][gateIndex][i];
            }

        }//end of tournament iteration

        //Now, tournament selection is over. We can move elites to new population. Red carpet ceremony begins...
        for (int popIndex=populationSize-popCountElite;popIndex<populationSize;popIndex++)
        {
            //Initialize variables
            gateCountAfterTournament[popIndex] = gateCountSorted[popIndex];
            popBitsAfterTournament[popIndex] = new bool*[gateCountSorted[popIndex]];
            popGateTypeAfterTournament[popIndex] = new int[gateCountSorted[popIndex]];

            //Move elites to new pop
            //Iterate through each gate
            for (int gateIndex=0;gateIndex<gateCountAfterTournament[popIndex];gateIndex++)
            {
                //Define new array for popBits
                popBitsAfterTournament[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];
                //Get gateType from old index
                popGateTypeAfterTournament[popIndex][gateIndex] = popGateTypeSorted[popIndex][gateIndex];
                //Now let's move previous popBits to popBitsAfterTournament
                for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                    popBitsAfterTournament[popIndex][gateIndex][i] = popBitsSorted[popIndex][gateIndex][i];
            }

        }

        //Let's start crossover! Nature wants reproduction of species!
        //First, we will create new population variables
        bool **popBitsAfterCrossover[populationSize];
        int *popGateTypeAfterCrossover[populationSize];
        int gateCountAfterCrossover[populationSize];

        //Start iterating over each population member and check if it will be crossed or not (we will exclude elites)
        for (int popIndex=0;popIndex<populationSize-popCountElite-1;popIndex++)
        {
            //Generate a random number (to be used for crossover probability)
            double r = ((double) rand() / (RAND_MAX));
            //Check if pop member will be crossed
            if (r<crossoverRate)
            {
                //Determine how many gates will be crossed
                //We will take the first pop member and cross with the consecutive one
                //Half of the first pop will be subject to crossover
                //However, if number of gates of the half of the first pop is fewer than consecutive one, we will only cross gateCount of the consecutive one
                int gateCountA = gateCountAfterTournament[popIndex];
                int gateCountB = gateCountAfterTournament[popIndex+1];

                int halfGateCountA = (int)((gateCountA/2));
                if (halfGateCountA==0) halfGateCountA=1;

                int gateCountToBeCrossed;
                if (halfGateCountA<=gateCountB) gateCountToBeCrossed = halfGateCountA;
                else gateCountToBeCrossed = gateCountB;

                //Initialize variables for new Population
                gateCountAfterCrossover[popIndex] = gateCountAfterTournament[popIndex];
                popBitsAfterCrossover[popIndex] = new bool*[gateCountAfterTournament[popIndex]];
                popGateTypeAfterCrossover[popIndex] = new int[gateCountAfterTournament[popIndex]];

                //Let's move the contents for the gates to be exchanged
                for (int gateIndex=0;gateIndex<gateCountToBeCrossed;gateIndex++)
                {
                    //Initialize array for popBits
                    popBitsAfterCrossover[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];

                    //Let's decide if gateType will be affected
                    double rForGateType = ((double) rand() / (RAND_MAX));
                    if (rForGateType<gateTypeCrossoverRate)
                    {
                        //Get gateType from consecutive pop
                        popGateTypeAfterCrossover[popIndex][gateIndex] = popGateTypeAfterTournament[popIndex+1][gateIndex];
                    }
                    else {
                        //Get gateType without change
                        popGateTypeAfterCrossover[popIndex][gateIndex] = popGateTypeAfterTournament[popIndex][gateIndex];
                    }

                    //Now let's move popBits to popBits new (move contents of the consecutive pop)
                    for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                        popBitsAfterCrossover[popIndex][gateIndex][i] = popBitsAfterTournament[popIndex+1][gateIndex][i];
                }

                //Let's fill the rest (remaining gates to new population)
                for (int gateIndex=gateCountToBeCrossed;gateIndex<gateCountAfterCrossover[popIndex];gateIndex++)
                {
                    //Initialize array for popBits
                    popBitsAfterCrossover[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];

                    //Get gateType without change
                    popGateTypeAfterCrossover[popIndex][gateIndex] = popGateTypeAfterTournament[popIndex][gateIndex];

                    //Now let's move popBits to popBits new (no change)
                    for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                        popBitsAfterCrossover[popIndex][gateIndex][i] = popBitsAfterTournament[popIndex][gateIndex][i];
                }
            }
            else //No crossover
            {
                //Initialize array
                gateCountAfterCrossover[popIndex] = gateCountAfterTournament[popIndex];
                popBitsAfterCrossover[popIndex] = new bool*[gateCountAfterTournament[popIndex]];
                popGateTypeAfterCrossover[popIndex] = new int[gateCountAfterTournament[popIndex]];

                //Let's move the contents without any change (all gates)
                for (int gateIndex=0;gateIndex<gateCountAfterCrossover[popIndex];gateIndex++)
                {
                    //Initialize array for popBits
                    popBitsAfterCrossover[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];

                    //Get gateType without change
                    popGateTypeAfterCrossover[popIndex][gateIndex] = popGateTypeAfterTournament[popIndex][gateIndex];

                    //Now let's move popBits to popBits new (directly)
                    for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                        popBitsAfterCrossover[popIndex][gateIndex][i] = popBitsAfterTournament[popIndex][gateIndex][i];
                }
            }//End of no crossover condition
        }//end popIndex for Crossover Operation

        //After crossover, move last pop and elite pops directly
        for (int popIndex=populationSize-popCountElite-1;popIndex<populationSize;popIndex++)
        {
            //Initialize variables for new Population
            gateCountAfterCrossover[popIndex] = gateCountAfterTournament[popIndex];
            popBitsAfterCrossover[popIndex] = new bool*[gateCountAfterTournament[popIndex]];
            popGateTypeAfterCrossover[popIndex] = new int[gateCountAfterTournament[popIndex]];

            //Let's move the contents without any change
            for (int gateIndex=0;gateIndex<gateCountAfterCrossover[popIndex];gateIndex++)
            {
                //Initialize array for popBits
                popBitsAfterCrossover[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];

                //Get gateType without change
                popGateTypeAfterCrossover[popIndex][gateIndex] = popGateTypeAfterTournament[popIndex][gateIndex];

                //Now let's move popBits to popBits new (directly)
                for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                    popBitsAfterCrossover[popIndex][gateIndex][i] = popBitsAfterTournament[popIndex][gateIndex][i];
            }
        }


        //Let's start mutation! (Sounds like a bad thing but it's good for the species...)

        //Start iterating over each population member and check if it will be mutated or not (we will exclude elites)
        for (int popIndex=0;popIndex<populationSize-popCountElite;popIndex++)
        {

            //Initialize variables for new Population (after mutation variables)
            gateCountAfterMutation[popIndex] = gateCountAfterCrossover[popIndex];
            popBitsAfterMutation[popIndex] = new bool*[gateCountAfterMutation[popIndex]];
            popGateTypeAfterMutation[popIndex] = new int[gateCountAfterMutation[popIndex]];

            //Generate a random number (to be used for mutation probability)
            double r = ((double) rand() / (RAND_MAX));
            //Check if pop member will be mutated
            if (r<mutationRate)
            {
                //Let's iterate through gates and apply mutation if needed
                for (int gateIndex=0;gateIndex<gateCountAfterMutation[popIndex];gateIndex++)
                {
                    //Randomly select gates for mutation
                    double rForMutation = ((double) rand() / (RAND_MAX));
                    if (rForMutation<gateMutationRate)
                    {
                        //Randomly assign a new gate type
                        popGateTypeAfterMutation[popIndex][gateIndex] = random(0,maxGateType);

                        //Fill the bits randomly
                        //Let's start randomly filling gate masks
                            //The length of array is number of inputs+gate index (gateindex+numberOfInputBits)
                            popBitsAfterMutation[popIndex][gateIndex]=new bool [gateIndex + numberOfInputBits];
                            //Generate a random number for connections count
                            int connectionsCount = random(2, connectMax);
                        //Fill connection times true
                        if ((gateIndex+numberOfInputBits)<=connectionsCount)
                        {
                            for (int i=0;i<(numberOfInputBits+gateIndex);i++) popBitsAfterMutation[popIndex][gateIndex][i]=true;
                        }
                            //Fill the rest with false
                        else {
                            for (int i=0;i<connectionsCount;i++) popBitsAfterMutation[popIndex][gateIndex][i]=true;
                            for (int i=connectionsCount; i < (gateIndex + numberOfInputBits); i++) popBitsAfterMutation[popIndex][gateIndex][i]=false;
                        }
                            //Now, shuffle'em! There you have a mutated gate with random connections...
                            randomShuffle(popBitsAfterMutation[popIndex][gateIndex], gateIndex + numberOfInputBits);//Shuffle contents
                    }//End mutate gate
                    else //Directly move gate (no mutation)
                    {
                        popBitsAfterMutation[popIndex][gateIndex]=new bool [gateIndex + numberOfInputBits];
                        //Get gateType without change
                        popGateTypeAfterMutation[popIndex][gateIndex] = popGateTypeAfterCrossover[popIndex][gateIndex];
                        //Now let's move popBits to popBits new (after mutation)
                        for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                            popBitsAfterMutation[popIndex][gateIndex][i] = popBitsAfterCrossover[popIndex][gateIndex][i];
                    }
                }//end gateindex
            }
            else //No Mutation
            {
                //Let's move the contents without any change (all gates)
                for (int gateIndex=0;gateIndex<gateCountAfterMutation[popIndex];gateIndex++)
                {
                    //Initialize array for popBits
                    popBitsAfterMutation[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];

                    //Get gateType without change
                    popGateTypeAfterMutation[popIndex][gateIndex] = popGateTypeAfterCrossover[popIndex][gateIndex];

                    //Now let's move popBits to popBits new (directly)
                    for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                        popBitsAfterMutation[popIndex][gateIndex][i] = popBitsAfterCrossover[popIndex][gateIndex][i];

                }
            }//End of no mutation condition
        }//end popIndex for Mutation Operation

        //After mutation, move elite pops directly (keep them secure!)
        for (int popIndex=populationSize-popCountElite;popIndex<populationSize;popIndex++)
        {
            //Initialize variables for new Population
            gateCountAfterMutation[popIndex] = gateCountAfterCrossover[popIndex];
            popBitsAfterMutation[popIndex] = new bool*[gateCountAfterMutation[popIndex]];
            popGateTypeAfterMutation[popIndex] = new int[gateCountAfterMutation[popIndex]];

            //Let's move the contents without any change
            for (int gateIndex=0;gateIndex<gateCountAfterMutation[popIndex];gateIndex++)
            {
                //Initialize array for popBits
                popBitsAfterMutation[popIndex][gateIndex] = new bool [numberOfInputBits + gateIndex];

                //Get gateType without change
                popGateTypeAfterMutation[popIndex][gateIndex] = popGateTypeAfterCrossover[popIndex][gateIndex];

                //Now let's move popBits to popBits new (directly)
                for (int i=0;i<(gateIndex + numberOfInputBits); i++)
                    popBitsAfterMutation[popIndex][gateIndex][i] = popBitsAfterCrossover[popIndex][gateIndex][i];
            }
        }

    }//end loop for GA iterations

    //We come to the end of GA loop. Now, we have the valuable chromosomes stored at popBitsAfterMutation.
    //If not terminated, we apply the same process for these chromosomes again and we will obtain a new generation
    //Every generation is expected to be better than the previous one!
    //Now, let's see the results...

    printf("\n");

    //Print Overall result
    printf("\n*****************SUMMARY*****************\n");
    printf("Population Count: %d\n",populationSize);
    printf("Tournament ratio: %.2f\n",tournamentRate);
    printf("Probability of crossover: %.2f\n",crossoverRate);
    printf("Probability of mutation: %.2f\n", mutationRate);
    printf("Elitism ratio: %.2f\n",elitismRate);
    printf("Augmentation Ratio: %.2f\n",augmentationPopRate);
    printf("Augmentation Speed: %.6f\n",augmentationRate);
    printf("Max Connections: %d\n",connectMax);
    printf("Use XOR: ");if (useXOR) printf("Yes\n");else printf("No\n");
    printf("CV Splits: %d\n",numberOfK);
    printf("Current CV Fold: %d\n",currentKFold);
    printf("Seed: %d\n",seed);

    printf("\n*************** OVERALL RESULT **********\n");
    printf("Best found on iteration: %d\n",bestTestAccIteration);
    printf("Population member: %d\n",bestTestAccIndex);
    printf("Train accuracy: %f\n",popAcc[bestTestAccIndex]);
    printf("Test accuracy: %f\n",bestTestAcc);
    printf("Best TP: %d\n",bestTP);
    printf("Best TN: %d\n",bestTN);
    printf("Best FP: %d\n",bestFP);
    printf("Best FN: %d\n",bestFN);
    printf("Recall (Sensitivity-TPR): %.8f\n",bestRecall);
    printf("Precision - PPV: %.8f\n",bestPrecision);
    printf("Specificity - TNR: %.8f\n",bestSpecificity);
    printf("False Positive Rate (FPR) (1-Specificity): %.8f\n",bestFalsePositiveRate);
    printf("F1-Score: %.8f\n",bestF_Score);
    printf("Balanced Accuracy: %.8f\n",bestBalancedAccuracy);
    printf("Gate count for best pop: %d\n",gateCount[bestTestAccIndex]);
    printf("--------------------------------------------------\n");
    for (int gIndex=0;gIndex<gateCount[bestTestAccIndex];gIndex++)
    {
        printf("Gate: %d",gIndex);
        if (gIndex==gateCount[bestTestAccIndex]-1) printf(" (Output gate)");
        printf("\n");
        printf("Gate type: %d\n",popGateType[bestTestAccIndex][gIndex]);
        printf("Gate connections:\n");
        printArray(popBits[bestTestAccIndex][gIndex], gIndex + numberOfInputBits);
    }
    printf("\n****************************** Test Acc History *****************************\n");
    for (int accH=0;accH<iterationCount;accH++)
        printf("%.12f,",testAccOnEachIteration[accH]);
    printf("\n****************************** Train Acc History *****************************\n");
    for (int accH=0;accH<iterationCount;accH++)
        printf("%.12f,",trainAccOnEachIteration[accH]);

    return 0;
}

//Here are some useful functions used in the code.

//Generates a random integer
int random(int from, int to){

    return rand() % (to - from + 1) + from;
}

//Finds the max value inside an array
int findMax(float arr[], int n, int gateCount[], int popIndex[])
{
    float max=0;
    int maxIndex=0;
    for (int i=0;i<n;i++)
    {
        if (arr[i]>max) {
            max=arr[i];
            maxIndex=i;
        }
        else if (arr[i]==max)
        {
            if (gateCount[popIndex[i]]<gateCount[popIndex[maxIndex]])
            {
                maxIndex=i;
            }
        }
    }
    return maxIndex;
}

//Logical OR gate with a given connections mask
bool getOrMask(bool maskArray[], bool outputArray[], int n)
{
    bool result = false;
    for (int i=0;i<n;i++)
    {
        if (maskArray[i] && outputArray[i]) result= true;
    }
    return result;
}

//Logical NOR gate with a given connections mask
bool getNOrMask(bool maskArray[], bool outputArray[], int n)
{
    bool result = false;
    for (int i=0;i<n;i++)
    {
        if (maskArray[i] && outputArray[i]) result= true;
    }
    return !result;
}

//Logical AND gate with a given connections mask
bool getAndMask(bool maskArray[], bool outputArray[], int n)
{
    bool result = true;
    for (int i=0;i<n;i++)
    {
        if (maskArray[i] && !outputArray[i]) result= false;
    }
    return result;
}

//Logical NAND gate with a given connections mask
bool getNAndMask(bool maskArray[], bool outputArray[], int n)
{
    bool result = true;
    for (int i=0;i<n;i++)
    {
        if (maskArray[i] && !outputArray[i]) result= false;
    }
    return !result;
}

//Logical XOR gate with a given connections mask
bool getXOrMask(bool maskArray[], bool outputArray[], int n)
{
    int numberOfTrue=0;
    for (int i=0;i<n;i++)
    {
        if (maskArray[i] && outputArray[i]) numberOfTrue++;
    }
    if ((numberOfTrue % 2)==0) return false;
    else return true;
}

//Logical XNOR gate with a given connections mask
bool getXNOrMask(bool maskArray[], bool outputArray[], int n)
{
    int numberOfTrue=0;
    for (int i=0;i<n;i++)
    {
        if (maskArray[i] && outputArray[i]) numberOfTrue++;
    }
    if ((numberOfTrue % 2)==0) return true;
    else return false;
}

//Swap two boolean values in memory
void swap (bool *a, bool *b)
{
    bool temp = *a;
    *a = *b;
    *b = temp;
}

//Same for an integer
void swapInt (int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

//Shuffle gate connections
void randomShuffle (bool arr[], int n)
{
    // Use a different seed value so that
    // we don't get same r esult each time
    // we run this program

    // Start from the last element and swap
    // one by one. We don't need to run for
    // the first element that's why i > 0
    for (int i = n - 1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        int j = rand() % (i + 1);

        // Swap arr[i] with the element
        // at random index
        swap(&arr[i], &arr[j]);
    }
}

//Shuffle integers
void randomShuffleInt (int arr[], int n)
{

    // Start from the last element and swap
    // one by one. We don't need to run for
    // the first element that's why i > 0
    for (int i = n - 1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        int j = rand() % (i + 1);

        // Swap arr[i] with the element
        // at random index
        swapInt(&arr[i], &arr[j]);
    }
}

//Prints gate connections
void printArray (bool arr[], int n)
{
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

//Converts boolean to string (true or false)
inline const char * const BoolToString(bool b)
{
    return b ? "true" : "false";
}


