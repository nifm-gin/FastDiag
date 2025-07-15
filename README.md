

## Codebase for the FastDiag I paper: 
### Is Less More? Comparing the Performance of Pauci to Multiparameter Prediction of the Neurotrauma Patient Pathway – a Comparative Observational Study (FASTDIAG I)

---
### Summary


### Setup
- Use python version 3.9+
- run ```pip install -r requirements.txt```

### Run the model computations to get all the scores
```python3 compute_models.py```

### Structure of the project
- `compute_models.py` is the main file which processes the data, runs hyperparameter search and returns metrics with confidence intervals for each model.
- `data/` contains the unprocessed anonymized database with all clinical variables and CT scan segmentation volumes for each patient as well as the outcomes.
- `old drafts/` contains old exploratory studies in notebook format.
- `results_summary/` contains the output csv files with performance metrics and 95% confidence intervals.
- `utility/` contains tool scripts.
