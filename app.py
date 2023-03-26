from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS, cross_origin
import os

import torch
from torch import nn
from transformers import BertTokenizer, BertModel

app = Flask(__name__, static_folder='build')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


labels = {
  0: 'stackoverflow.com',
  1: 'lifehacks.stackexchange.com',
  2: 'literature.stackexchange.com',
  3: 'english.stackexchange.com',
  4: 'philosophy.stackexchange.com',
  5: 'monero.stackexchange.com',
  6: 'softwareengineering.stackexchange.com',
  7: 'vi.stackexchange.com',
  8: 'health.stackexchange.com',
  9: 'quant.stackexchange.com',
  10: 'worldbuilding.stackexchange.com',
  11: 'gardening.stackexchange.com',
  12: 'ai.stackexchange.com',
  13: 'devops.stackexchange.com',
  14: 'astronomy.stackexchange.com',
  15: 'french.stackexchange.com',
  16: 'linguistics.stackexchange.com',
  17: 'opendata.stackexchange.com',
  18: 'woodworking.stackexchange.com',
  19: 'webapps.stackexchange.com',
  20: 'patents.stackexchange.com',
  21: 'joomla.stackexchange.com',
  22: 'magento.stackexchange.com',
  23: 'stats.stackexchange.com',
  24: 'spanish.stackexchange.com',
  25: 'parenting.stackexchange.com',
  26: 'photo.stackexchange.com',
  27: 'pets.stackexchange.com',
  28: 'codereview.stackexchange.com',
  29: 'networkengineering.stackexchange.com',
  30: 'politics.stackexchange.com',
  31: 'tridion.stackexchange.com',
  32: 'eosio.stackexchange.com',
  33: 'rpg.stackexchange.com',
  34: 'cooking.stackexchange.com',
  35: 'dba.stackexchange.com',
  36: 'arduino.stackexchange.com',
  37: 'bioinformatics.stackexchange.com',
  38: 'blender.stackexchange.com',
  39: 'diy.stackexchange.com',
  40: 'graphicdesign.stackexchange.com',
  41: 'bitcoin.stackexchange.com',
  42: 'elementaryos.stackexchange.com',
  43: 'civicrm.stackexchange.com',
  44: 'gaming.stackexchange.com',
  45: 'tezos.stackexchange.com',
  46: 'mathematica.stackexchange.com',
  47: 'unix.stackexchange.com',
  48: 'freelancing.stackexchange.com',
  49: 'physics.stackexchange.com',
  50: 'craftcms.stackexchange.com',
  51: 'anime.stackexchange.com',
  52: 'pm.stackexchange.com',
  53: 'serverfault.com',
  54: 'salesforce.stackexchange.com',
  55: 'reverseengineering.stackexchange.com',
  56: 'hardwarerecs.stackexchange.com',
  57: 'christianity.stackexchange.com',
  58: 'emacs.stackexchange.com',
  59: 'windowsphone.stackexchange.com',
  60: 'raspberrypi.stackexchange.com',
  61: 'electronics.stackexchange.com',
  62: 'codegolf.stackexchange.com',
  63: 'money.stackexchange.com',
  64: 'askubuntu.com',
  65: 'superuser.com',
  66: 'android.stackexchange.com',
  67: 'german.stackexchange.com',
  68: 'bicycles.stackexchange.com',
  69: 'matheducators.stackexchange.com',
  70: 'korean.stackexchange.com',
  71: 'buddhism.stackexchange.com',
  72: 'islam.stackexchange.com',
  73: 'sports.stackexchange.com',
  74: 'travel.stackexchange.com',
  75: 'fitness.stackexchange.com',
  76: 'hermeneutics.stackexchange.com',
  77: 'history.stackexchange.com',
  78: 'russian.stackexchange.com',
  79: 'law.stackexchange.com',
  80: 'opensource.stackexchange.com',
  81: 'chess.stackexchange.com',
  82: 'skeptics.stackexchange.com',
  83: 'puzzling.stackexchange.com',
  84: 'ham.stackexchange.com',
  85: 'academia.stackexchange.com',
  86: 'sqa.stackexchange.com',
  87: 'engineering.stackexchange.com',
  88: 'space.stackexchange.com',
  89: 'avp.stackexchange.com',
  90: 'economics.stackexchange.com',
  91: 'homebrew.stackexchange.com',
  92: 'italian.stackexchange.com',
  93: 'music.stackexchange.com',
  94: 'substrate.stackexchange.com',
  95: 'hinduism.stackexchange.com',
  96: 'computergraphics.stackexchange.com',
  97: 'robotics.stackexchange.com',
  98: 'expressionengine.stackexchange.com',
  99: 'drupal.stackexchange.com',
  100: 'softwarerecs.stackexchange.com',
  101: 'sitecore.stackexchange.com',
  102: '3dprinting.stackexchange.com',
  103: 'chemistry.stackexchange.com',
  104: 'expatriates.stackexchange.com',
  105: 'poker.stackexchange.com',
  106: 'writers.stackexchange.com',
  107: 'datascience.stackexchange.com',
  108: 'gis.stackexchange.com',
  109: 'quantumcomputing.stackexchange.com',
  110: 'ux.stackexchange.com',
  111: 'ell.stackexchange.com',
  112: 'japanese.stackexchange.com',
  113: 'earthscience.stackexchange.com',
  114: 'sound.stackexchange.com',
  115: 'boardgames.stackexchange.com',
  116: 'apple.stackexchange.com',
  117: 'dsp.stackexchange.com',
  118: 'scicomp.stackexchange.com',
  119: 'cardano.stackexchange.com',
  120: 'mathoverflow.net',
  121: 'cstheory.stackexchange.com',
  122: 'wordpress.stackexchange.com',
  123: 'hsm.stackexchange.com',
  124: 'biology.stackexchange.com',
  125: 'math.stackexchange.com',
  126: 'scifi.stackexchange.com',
  127: 'gamedev.stackexchange.com',
  128: 'workplace.stackexchange.com',
  129: 'ethereum.stackexchange.com',
  130: 'security.stackexchange.com',
  131: 'cs.stackexchange.com',
  132: 'rus.stackexchange.com',
  133: 'solana.stackexchange.com',
  134: 'outdoors.stackexchange.com',
  135: 'latin.stackexchange.com',
  136: 'aviation.stackexchange.com',
  137: 'crypto.stackexchange.com',
  138: 'retrocomputing.stackexchange.com',
  139: 'movies.stackexchange.com',
  140: 'webmasters.stackexchange.com',
  141: 'tex.stackexchange.com',
  142: 'cogsci.stackexchange.com',
  143: 'judaism.stackexchange.com',
  144: 'sharepoint.stackexchange.com',
  145: 'tor.stackexchange.com',
  146: 'chinese.stackexchange.com',
  147: 'mechanics.stackexchange.com'
}

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 148)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_id, mask):
        x = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        x = self.dropout(x[1])
        x = self.linear(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x

    def inference(self, input_id, mask, top_k):
        x = self.forward(input_id, mask)
        x = torch.topk(x, k=top_k, dim=1)
        return x


@app.route('/api/predict', methods=['POST'])
@cross_origin()
def predict():
  model = BertClassifier()
  model.load_state_dict(torch.load('./training/model.pt', map_location=torch.device('cpu')))
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

  # input_text = request.form['text']
  input_text = request.get_json()['title']

  top_k = 10
  tokens = tokenizer(input_text)
  input_ids = torch.tensor([tokens['input_ids']])
  mask = torch.tensor([tokens['attention_mask']])
  prediction = model.inference(input_ids, mask, top_k)

  forums = [labels[i] for i in prediction.indices[0].tolist()]
  probabilities = prediction.values[0].tolist()

  rates = [
    {
      'forum': forums[i],
      'probability': probabilities[i]
    }
     for i in range(top_k)
  ]

  return jsonify({'rates': rates})

