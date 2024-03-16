<br>
uploading python file for the assignment<br>
I have a few queries/doubts though<br>
1]The APIs given do not change any data on minute level granularity. Why promise changes if there are none?<br>
2]Clearly the minute level changes are in premium prices, I feel like we were supposed to use those to predict straddle<br>
3]However, no API was provided to obtain access to premium prices<br>
Therefore what I did was simply use the prices of today to predict straddle prices of tomorrow<br>
Is this a good solution? no. Clearly they're also influenced by premium prices however without access to that data, I feel like this is the best I could do.<br>
I used a simple linear regression model (not even regularised) to simplify calculations as I am not too sure of the best estimator model to use.<br>
Nonetheless I feel like my attempt has been decent and am uploading it<br>
 
