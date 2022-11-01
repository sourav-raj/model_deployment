### DL Model Deployment with tensorflow js 

- In this we direcly deploy model on client side (webpage, mobile etc.) to 
    1) speed up the processing
    2) removing dependency on backend server
    3) to avoid privacy concern

- For this first we will export trained model then load the same model with javascipt. Alternatively, we can also build/create model using tf.js

- first install tensforflowjs
    pip install tensorflowjs

- save trained model with tfjs coverters
    1) import tensorflowjs as tfjs
    2) tfjs.converters.save_keras_model(model, "/content/")

    3) put model.json & get group.bin file from content to a folder 
    4) create tfjsclient.html and add tensorflow.js in script

        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>

    5) add function in body

        <body>
        <script>
            async function loadModel(){
                const model = await tf.loadLayersModel('model.json'); 
                model.predict(tf.tensor2d([[-1.40, -0.817]])).print()   
            }       
            loadModel()

        </script>
    </body>1)

    Here async make sure page will be loaded even this line execution not completed & once complete will show the o/p.




