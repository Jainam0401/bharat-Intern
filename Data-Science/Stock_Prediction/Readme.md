# LSTM Stock Forecasting from Microsoft

This Python notebook shows how to anticipate the stock price of Microsoft (MSFT) based on historical data using Long Short-Term Memory (LSTM) neural networks. TensorFlow and Pandas libraries are used in the notebook for modelling and data processing.

## General

Based on previous stock data, the goal of this research is to forecast Microsoft's stock price. On the basis of historical stock price data, we will train an LSTM neural network to forecast future dates.

Dataset ##

Yahoo Finance is used to get Microsoft's historical stock data. The dataset includes details on each trading day's trade volume, date, opening price, highest price, lowest price, and closing price as well as their adjusted closing prices.

## Steps

1. **Data Preparation**: The historical stock data must first be loaded and processed. The 'Date' and 'Close' columns are taken out of the dataset, and the 'Date' column is then converted to a datetime format.

2. **Windowed Data Generation**: In order to train the LSTM model, we must generate a windowed dataset, where each data point is made up of a list of previous "Close" prices with the next "Close" price as the goal. To create the windowed dataset, we define the method 'df_to_windowed_df()'.

3. **Model Architecture**: Using the TensorFlow Keras API, we build an LSTM-based neural network. An input layer, an LSTM layer with 64 units, two Dense layers each with 32 units and the "relu" activation function, and an output Dense layer are all components of the model.

4. **Model Compilation**: The model is put together using Adam optimizer with a learning rate of 0.001 and Mean Squared Error (MSE) as the loss function. During training, we additionally monitor the mean absolute error as a measure.

5. **Model Training**: Using the 'fit' function, the LSTM model is trained on the windowed dataset. 100 epochs are used for training.

6. **Model Evaluation**: We assess the model's functionality using the training, validation, and testing datasets after training. To see the model's predictions in action, we compare the anticipated stock prices to the actual stock prices.

7. **Recursive Predictions**: Finally, we use the trained model to make recursive predictions. By updating the most recent data window with each subsequent prediction, we iteratively forecast future share values. As a result, we can predict stock values outside of the test dataset.

## Recommendations

You must have Python and Jupyter Notebook installed in order to use this notebook. You may use the following command to install the necessary libraries:

Run the command "bash pip install pandas numpy matplotlib tensorflow"

Next, save the "MSFT.csv" dataset to the same directory as this notebook by clicking on the supplied link.

Run the notebook cell-by-cell to observe the LSTM stock forecasting process in detail.

## Summary

This notebook gives an example of how to forecast stock prices using LSTMs. We developed a model using previous data and applied it to forecast future dates. Remember that making forecasts about the stock market involves risk, therefore it is not advisable to rely all of your financial decisions on them. However, the methods examined in this notebook provide useful information about applying deep learning to time-series forecasting challenges.

