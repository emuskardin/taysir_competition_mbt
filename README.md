## TAYSIR Competition - Black-box testing oriented approach

This repository contains all data required to reproduce models extracted from RNNs and transformers during TAYSIR competition.
All models were submitted under username `EdiMuskardin`. 

This work is joint collaboration between Edi Muskardin and Martin Tappler.

Repository Structure
- extraction_classifier.py - used to extract models for Track 1
- extraction_regression.py - used to extract models for Track 2. Contains an abstraction mapper.
- SULs.py - SUL classes for all types of networks
- utils.py - utility functions
- compact_submission.py - helper file used to create compact submissions

## Track 1 - Binary Classification

### Result Summary

| Dataset | Error Rate | Model Size | Notes |
|---------|------------|------------|-------|
| 1       | 0.075      | 7700         |    most likely context free   |
| 2       | 0          | 8         |   regular    |
| 3       | 0          | 9         |   regular    |
| 4       | 0          | 9         |   regular    |
| 5       | 0          | 5         |    regular   |
| 6       | 0.0002     | 18         |   can find many cex with strong oracle    |
| 7       | 0          | 2         |   transformer, 2 states    |
| 8       | 0.32       | Unrelated to accuracy         |     Unknown structure  |
| 9       | 0.007      | [500-1500]         |    No regular representation   |
| 10      | 0.014      | 1500         |     No regular representation  |
| 11      | 0.007       | 500         |   Consider only len(val) < 100    |


## Track 2 - Language Modelling/Regression

### Result Summary

| Dataset | Error Rate | Model Size | Learning Parameters |
|---------|------------|------------|---------------------|
| 1       | 0.175      | 860         |    500 LR, 200 bins                 |
| 2       | 0.0097     | 123         |          100 LR, 10 bins           |
| 3       | 0.00003     | 106         |     50 LR, 10 bins                |
| 4       | 0.000006   | 102         |        50LR, 10 bins            |
| 5       | 0.00000007 | 118         |         50 LR, 12 bins           |
| 6       | 0.1971      | 318         |      200 LR, 20 bins               |
| 7       | 0.0      | 150         |       100 LR, 15 bins        |
| 8       | 0.0443     | 146         |     100 LR,   15 bins           |
| 9       | 0.0        | 55         | 30 LR, 10 bins      |
| 10      | 0.1237      | 1412         | 200 LR, 20 bins           |