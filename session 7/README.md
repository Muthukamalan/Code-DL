Objective
===
    
    1. 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
    2. Less than or equal to 15 Epochs
    3. Less than 8000 Parameters


`Note`:

    In earlier session, we use SE architecture with 20K params so I'm stick to that architecture and moving on from there and stick with default learning rate

## Lessons Learned from Delta Updates

I. Model_001

    CONFIG:

- [ ] apply train transformation

- [ ] batchnorm 

- [ ] dropout

- [X] 15 epoch
    
    TARGETS:
        
- under 8K params with $50^{+}\delta$%

    RESULTS:

- test accuracy were stuck with 10% even If i extend epoch 25 with 7415 params

    ANALYSIS:

- Neural Network not at all learning       



II. Model_002

    CONFIG:

- [ ] apply train transformation

- [X] no batchnorm 

- [ ] no dropout

- [X] 15 epoch

    TARGETS:
        
- under 8K params with $50^{+}\delta$%

    RESULTS:

- results were showing 90.2% consistance accuracy in test and in train 99.4 with 7645 params

    ANALYSIS:

- adding batch norm to batch norm gives exponential returns


III. Model_003

    CONFIG:

- [ ] apply train transformation

- [X] batchnorm 

- [X] dropout=0.01

- [X] 15 epoch

    TARGETS:
        
- adding dropouts to see increase in accuracy

    RESULTS:

- results were surprising! train=99.19 and test=99.27 with 7645 params

    ANALYSIS:

- actually adding dropout results in decrease in test accuracy but slight improvement in test accuracy
- earlier train=99.4 and test=99.2 and this time train=99.19 and test=99.27



VI. Model_006

    CONFIG:

- [ ] apply train transformation

- [X] batchnorm 

- [X] dropout=0.1

- [X] 15 epoch

    TARGETS:
        
- adding more dropouts to see if any improvement in accuracy

    RESULTS:

- results were surprising again! train=97.32 and test=98.5 with 7645 params

    ANALYSIS:

- actually adding more dropouts would leads to bad for neural network.




VII. Model_007

    CONFIG:

- [X] apply train transformation

- [X] batchnorm 

- [X] dropout=0.01

- [X] 15 epoch

    TARGETS:
        
- adding augmentation technique

    RESULTS:

- results were achieved. train=98.79 and test=99.4 consistently

    ANALYSIS:

- adding augmentation helps to improve accuracy better