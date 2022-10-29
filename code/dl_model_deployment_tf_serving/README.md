### DL Model Deployment on Docker with tensorflow Serving 

- In this we first train a deep learning model. We save this model with tensorflow serving.

-  Tensorflow provide tf serving through which we can save & export our DL models.

    model.save("model/v1"): this contains 3 dir: assets, saved_model.pb & vaiables for all graphs, actual model in protobuf format & wt.,biases checkpoints resp.
    
- Once model is save we can simple use the saved model anywhere.

- Another method would be to use tensorflow serving to create a REST API in docker & client will call this based on http protocol.

    1) docker pull tensorflow/serving
    2) docker run -t --rm -p 8501:8501 -v "G:\git\model_deployment\data\saved_model\dl_model:/models/dl_model" -e MODEL_NAME=product_purchase_model  tensorflow/serving &
    
    
    

