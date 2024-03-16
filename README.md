# StatYun
\\
uploading python file for the assignment\\
I have a few queries/doubts though\\
1]The APIs given do not change any data on minute level granularity. Why promise changes if there are none?\\
2]Clearly the minute level changes are in premium prices, I feel like we were supposed to use those to predict straddle\\
3]However, no API was provided to obtain access to premium prices\\
Therefore what I did was simply use the prices of today to predict straddle prices of tomorrow\\
Is this a good solution? no. Clearly they're also influenced by premium prices however without access to that data, I feel like this is the best I could do.\\
I used a simple linear regression model (not even regularised) to simplify calculations as I am not too sure of the best estimator model to use.\\
Nonetheless I feel like my attempt has been decent and am uploading it\\
 
