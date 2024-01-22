# Marked-Hawkes-Estimation
Real Data contains the cryptocurrency data sets. filename- btcusd_1655337600 means BTC-USD market orders, 1655337600 denotes the time in epoch time. 
The dataset is in the order Order ID, Epoch timestamp, volume, price, buyer ID, seller ID, Whether buyer is market maker or not.
If Whether buyer is market maker or not is Y then its sell trade, else if it is N then buy trade.

In parametric folder for Marked Hawkes Process Estimation we use exponential kernel and exponential mark distribution with decoupled mark and time kernel function.

In Non-parametric folder for Marked Hawkes Estimation we have both SNH - univariate and multivariate Estimation models and NNNH- univariate model. 
