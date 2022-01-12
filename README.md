# DL Project

Instructions to reproduce our results:

- Chose a model from the list of branches. The branches that start with hand_in_ are the final version of the respective model.
- Download the dataset following the instructions provided on the [WeatherBench](https://github.com/pangeo-data/WeatherBench) github. We use only the 5.625deg version and only the temperature at 850 hPa.
A possible command would be:
```wget "https://dataserv.ub.tum.de/s/m1524895/download?path=%2F5.625deg%2Ftemperature_850&files=temperature_850_5.625deg.zip" -O temperature_850_5.625deg.zip```
- Extract the dataset (make sure it is unzipped correctly) and place it in a seperate folder.
- To start the training run ```python main.py --data_path your_custom_path``` replacing ```your_custom_path``` with the path to the folder you placed your data in.
- In case of memory issues reduce the batch size by passing ```--batch_size 1```

---


To additionally run the evaluation do:

```python main.py --load_checkpoint attention_10.pt --mode val```

---

This model inherently creates a iterative prediction. We have composed a video of the predicted frames in [movie.gif](movie.gif)
