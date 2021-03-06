import os
from flask import Flask, url_for, render_template, request, flash, redirect, send_from_directory, after_this_request
from werkzeug.utils import secure_filename
import secrets
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

secret = secrets.token_hex(16)
app.secret_key = secret
# File implementation taken from https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
UPLOAD_FOLDER = 'uploaded_files'
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#content length in bytes (24 mb)
app.config['MAX_CONTENT_LENGTH'] = 24 * 1024 * 1024
new_model = tf.keras.models.load_model('leaf_health_classififer3')

def file_allowed(file):
    if '.' in file and file.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS:
        return True
    else:
        return False


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/uploaded/<image>')
def return_image(image):
    #Return requested image (used in img src?)
    return send_from_directory(app.config['UPLOAD_FOLDER'],image)
    
@app.route('/classify',methods=['POST','GET'])
def classify():
    print('classify called')
    if request.method == 'GET':
        return render_template('classify_get.html')
    else:
        #Delete files after request
        print(os.listdir(app.config['UPLOAD_FOLDER']))
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'],file))
        if request.files['file'] == '':
            flash('No file was given!')
            return redirect(url_for('classify'))
        try:
            #Get the image filename
            print(request.files)
            image = request.files['file']
            print(image.filename)
            assert(file_allowed(image.filename))
            #Sanitize the filename to ensure no bad stuff happens
            #print(app.config)
            #path = (app.config['UPLOAD_FOLDER'] + '/' + filename)
            #print('attempting to save at',path)
            #image.save(path)
            #print('saved')
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


            #Formats image to be numpy array
            img = Image.open(image)
            img = img.resize((80,80))
            #May consider image processing to improve results...maybe
            #time will tell
            #perhaps mess with background? idk that is above pay grade

            img = [[np.array(img)]] #idk the brackets worked in vscode idk
            #print(img)
        except Exception as error:
            #If (aboutToCrash()):
            #   dont()
            print(error)
            print('invalid file')
            flash('Your file is invalid.')
            return redirect(url_for('classify'))
        # Do tensorflow stuff here :)
        #load Model
        #TODO: Find file name
        #new_model = tf.keras.models.load_model('leaf_health_classififer')
        # Check its architecture
        new_model.summary() 
        #predictions = new_model.predict(image)
        #print(predictions) #I actually don't know how extactly this is formatted, I know its a numpy array so we can scprit it, don't know where to do that so pls run this print statment thank you
        #no clue if it will work ethier so there is that one too
        #REMINDER: May have to copy above tensorflow code and do it on VSCode
        try:
            img_array = tf.reshape(img, (1, 80, 80, 3))
            #print(img_array.shape)
            predictions = new_model.predict(img_array)
            
        
            score = tf.nn.softmax(predictions[0])
            classed = np.argmax(score)
            confidence = np.max(score)
            #print(np.argmax(score), np.max(score))

            #results = [classed, confidence]
            results = classed
        except Exception as error:
            flash("We're sorry, we cannot process your image. Please use a different image.")
            return redirect(url_for('classify'))
        print('successfully handled')
        return render_template('classify_post.html',results=results,confidence=confidence,filename=filename)


@app.route('/explanation')
def explanation():
    return render_template('explanation.html')




#@app.route('/results')
#def results():
    #pass


    
if __name__=="__main__":
    app.run(host='0.0.0.0',threaded=True)
    

#[packages]
#gunicorn = "*"
#flask = "*"
#requests = "*"
#wtforms = "*"
#flask_assets = "*"
#flask_static_compress = #"*"
#tensorflow = "*"
#[dev-packages]#
