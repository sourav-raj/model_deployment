### Serverless ML API:

Using VM, we have to pay for the instance even if no one is using it. To avoid this, we use serverlesss architecture so that we don't have to worry about underlying cost of servers.

We focus on writing our business logic of functions/methods & we will be charged on no. of times out function is invoked.

1) first we need to store our trained model on cloud
    - use bucket in GCP / S3 in AWS/ blob storage in azure.
    - create bucket then create a folder then upload model files
    
2) create cloud function in GCP/lambda in AWS.
    - region oshould match with bucket region while creating
    - use trigger type as http
    - choose runtime python 3.7 & write our code then deploy.
3) there is a testing function which will take json input for prediction.
