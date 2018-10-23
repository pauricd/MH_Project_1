This project contains the following files:

Python Files:
    -Individual.py - Class for the Individual - Added new files for template given to us
    -TSP_toStudents.py - The implementation of the GA for this assignment.

Report:
    -Pauric_Dawson_r00169689_MetaheuristicOptimisation_Assignment1.pdf  - report of my finds and description ot GA operators.
    -AS1_3col.pdf - the 3col chart ina  pdf document in case it to small to read in the report
    -As1_3c0l_Sol.pdf- Coloured solution for the 3COL question, again it is int he report but just incase its needed for further examination
    -TruthTable_ASSIG1.xlsx - A truth table I used to prove the solution to the 3COl graph, again its in the report as well.

Data Set:
    -dataset directory - all teh inst<n>.tsp files

Output Results file:
    -Baseline.csv  - Output for the execution of the required configs 1-6. 3 instances executed * 6 Configs * 3 executions run
    -Iterations100.csv - Output of a test where Iteration was set to 100
    -Iterations600.csv - Output of a test where Iteration was set to 600
    -Mutation_point4.csv - Output of a test where the Mutation Rate was set to 0.4
    -Mutation_point_zero_zero_1.csv - Output of a test where the Mutation Rate was set to 0.001
    -output.csv - Default output file, currently the output will be put to this file
    -PopulationSize_30.csv - Output of a test where the Population Size was Set to 30
    -PopulationSize_400.csv - Output of a test where the Population Size was Set to 400

Enviroment:
   -Written in Python 3.7

Execution :
    -The program is self contained, if you run the python program now, it will run all configuration against the
    3 instances (ints0.txt, inst13.txt, inst16.txt) and it will run it just one time for each.

   -If you wish to run more runs (as per the assignment of 3) then edit the program
    at line 495, and assign a new value to  numberofrepeatingrun, currently it is set to 1.

   -If you would like to jsut run against one instance then comment line 426 )instances = ["dataset/inst-0.tsp","dataset/inst-13.tsp","dataset/inst-16.tsp"])
    and uncomment line 427, this will run inst-0.tsp, change it to waht ever instance you want.

   -If you want to use a different output file name (currently set to output.csv) the change it at line 474

   -The output is a cvs format, for easy import in excel or into python dataframes to further analysis.

   -Further consideration was to put these small configs(output file name, number of executions  and instance to run against) into a ymal file for easier config.














