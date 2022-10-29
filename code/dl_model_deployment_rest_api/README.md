### DL Model Deployment on Docker with tensorflow Serving 

- In this we first train a deep learning model. We save this model with tensorflow serving.

-  Tensorflow provide tf serving through which we can save & export our DL models.

    model.save("model/v1"): this contains 3 dir: assets, saved_model.pb & vaiables for all graphs, actual model in protobuf format & wt.,biases checkpoints resp.
    
- Once model is save we can simply use flask to create a REST API server & expose to client.


    
    
    

