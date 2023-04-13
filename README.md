## TAYSIR

## Track 1 - Binary Classification

| Dataset | Error Rate | Model Size | Notes |
|---------|------------|------------|-------|
| 1       | 0.075      | 7700         |    most likely context free   |
| 2       | 0          | 8         |   regular    |
| 3       | 0          | 9         |   regular    |
| 4       | 0          | 9         |   regular    |
| 5       | 0          | 5         |    regular   |
| 6       | 0.0002     | 18         |   can find many cex with strong oracle    |
| 7       | 0          | 2         |   transformer, 2 states    |
| 8       | 0.32       | inf         |     unknown grammar type  |
| 9       | 0.007      | xy         |       |
| 10      | 0.014      | 1500         |       |
| 11      | 0.29       | inf         |   unknown grammar type    |


## Track 2 - Language Modelling/Regression

| Dataset | Error Rate | Model Size | Learning Parameters | Notes |
|---------|------------|------------|---------------------|-------|
| 1       | 0.175      | xy         |                     |       |
| 2       | 0.0097     | xy         |                     |       |
| 3       | 0.00003     | xy         |                     |       |
| 4       | 0.000006   | xy         |                     |       |
| 5       | 0.00000007 | xy         |                     |       |
| 6       | 0.1971      | 318         |      200 LR, 20 bins               |       |
| 7       | 0.0      | 150         |       100 LR, 15 bins        |       |
| 8       | 0.0443     | xy         |                     |       |
| 9       | 0.0        | 55         | 30 LR, 10 bins      |       |
| 10      | 0.1237      | 1412         | 200 LR, 20 bins           |  transformer     |