# cellpose-bmz-wrapper
#### A wrapper to make a bioimage model zoo compatible package out of cellpose models  

The `model.py` contains the wrapper model's code which is a subclass of both `torch.nn.Module` and the `CellposeModel` classes. This model is based on cellpose `cyto3` model, and the `SizeModel` is also included.  
To produce sample input/outputs you can use the `data_preparation` notebbok. And to pack the model for the _BMZ_ use the `model_preparation_cellpose` notebook.

### Usage example
```python
model = CellPoseWrapper(estimate_diam=True)
model.load_state_dict(
    torch.load("./cellpose_models/cyto3", map_location=model.device)
)
masks, flows, styles, diams = model(img_batch)
```

### Outputs
This model provides four outputs:
- masks: an array of shape `b,y,x`
- flows: an array of shape `b,8,y,x`
    - For each input image flows are stacked together.
- styles: an array of shape `b,y,x`
- diams: the estimated diameter of shape `1`
