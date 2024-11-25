# cellpose-bmz-wrapper
#### A wrapper to make a bioimage model zoo compatible package out of Cellpose models  

The `model.py` contains the wrapper model's code which is a subclass of both `torch.nn.Module` and the `CellposeModel` classes. The `SizeModel` for `cyto3` and `nuclei` models are also included, so you can set `estimate_diam=True` to use the `SizeModel` to estimate the object diameter.   

To produce sample input/outputs you can use the `data_preparation` notebook. And to pack the model for the _BMZ_ use the `model_preparation_cellpose` notebook.

### Usage example
```python
model = CellPoseWrapper(model_type="cyto3", estimate_diam=True)
model.load_state_dict(
    torch.load("./cellpose_models/cyto3", map_location=model.device)
)
masks, flows, styles, diams = model(img_batch)
```

### Outputs
This model provides four outputs:
- masks: an array of shape `b,y,x`
- flows: an array of shape `b,6,y,x`
    - For each input image flows are stacked together.
- styles: an array of shape `b,256`
- diams: the estimated diameter of shape `b,1`
