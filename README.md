Forum finder takes in a question and outputs the best forums to ask the question in. 

To get set up, create a conda environment on python 3.10 (optional but recommended), then install the requirements with ```pip install -r requirements.txt```. 

To get the weights for the classifier, you can either 
1) get the data from https://drive.google.com/file/d/1sslXiMTOQs31ZF3uKJebgreHqHrrKTu4/view?usp=sharing, place the file into the training folder, and train the classifier with ```python classifier.py```, or
2) get pretrained weights from https://drive.google.com/file/d/1vlKzvMT-CMtIpm6fi3zgAT7BhwnTrPFr/view?usp=sharing and place the file into the training folder. 

Update any frontend changes with ```npm run build``` in the frontend folder. 

Run the server and web interface with ```flask run```. 

Check out https://sites.google.com/uw.edu/forum-finder/home for an overview and demo of this project!