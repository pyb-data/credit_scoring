# credit_scoring

All the different steps from data analysis to model training are contained in the jupyter notbooks (.ipynb)

The code to be deployed is embeddeed in projet_7.zip, it contains:
    - the code of the API (which returns either a JSON with a client scoring or an web page from which the model can be run and the dashboard can be launched)
    - the code of the dashboard
    - the code of the data preparation
    - a file projet_7 to be copied in /etc/nginx/sites-enabled/
    - a folder that contains some pickle objets (including the pipeline)
    - a folder that contains the html template for Flask
    
Two pdf are available: a presentation and a methodology (from data analysis to deployment on EC2)


