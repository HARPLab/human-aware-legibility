README:
ADANOTE is sprinkled in with most relevant commentary to someone editing the legibility code.

SETUP:
conda_setup.yaml contains a description of the required conda libs
Install this ilqr library to the same directory where you installed the main directory of this repo. Ie, should match the imports of ../../ilqr in this subdirectory.
https://github.com/anassinator/ilqr

RUNNING:
Run the file SandboxTestExperiments.py, and it will run all of the scenarios set up in SandboxTestScenarios.py
Scenarios include: start, goals, target, obstacles, viewers, number of steps to solve.

Experiments includes options like 'test_amount_of_slack', which varies the amount of slack for you to determine the best value.

IN PROGRESS/Misc notes:
Obstacles are close but not quite right, and the symmetricality of dist_sqr makes it hard for it to handle colinear cases, it seems.
If your n isn't enough, you won't get a path.
