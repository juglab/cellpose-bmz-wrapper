# cellpose-bmz-wrapper
#### A wrapper to pack cellpose model for bioimage model zoo  

The `model.py` contains the wrapper code which is basically a simple pytorch model that uses the _`cellpose`_ APIs to generate masks for the inputs.  
To produce sample input/outputs you can use the `data_preparation` notebbok. And to pack the model for the _BMZ_ use the `model_preparation_original_cellpose` notebook.

### Outputs
This model provides two outputs:
- Masks: an array of shape `b,y,x`
- Flows: an array of shape `b,8,y,x`
    - For each input image flows are stacked together.
