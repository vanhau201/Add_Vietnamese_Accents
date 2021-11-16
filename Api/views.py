from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np
import tensorflow as tf
import re
from nltk import ngrams
from collections import Counter
import unidecode
from rest_framework import status
# Create your views here.

model = tf.keras.models.load_model("./Model_Predict/model_ngram10_final_new.h5")

@api_view(['POST'])
def Predict(request):
    try :

        if request.method == 'POST':
            sent = request.data['data']
            pre = predict(sent)
        return Response(pre)
    except:
        return Response(status=status.HTTP_400_BAD_REQUEST)



# các chữ cái trong tiếng việt
alphabets = ' aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz0123456789/'
alphabets = [i for i in alphabets]

# hàm xử lí câu
def preprocess(sent):
  sent = re.sub(
      r'[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789/ ]',
      "", sent)
  sent = ' '.join([char.strip() for char in sent.split()])
  # sent = word_tokenize(sent,format='text')
  sent = sent.lower()
  return sent

# hàm tách ngram
def gen_ngrams(sent):
    if len(sent.split())<10:
        return [sent.split()]
    return ngrams(sent.split(), 10)
# hàm chuyển câu thành vector
def convert_to_vector(text, maxlen): 
  x = np.zeros((maxlen, len(alphabets)))
  for i, v in enumerate(text):
    x[i, alphabets.index(v)] = 1
  return x

# hàm chuyển vector sang câu
def convert_to_text(x):
  x = x.argmax(axis=-1)
  rs = ''.join(alphabets[i] for i in x)
  return rs.strip()

def predict(text):
    max_len = 65
    text = preprocess(text)
    text = unidecode.unidecode(text)
    ngram = list(gen_ngrams(text))
    ngram = [' '.join(i) for i in ngram]
    vector_ngram = np.array([convert_to_vector(i,max_len) for i in ngram])
    pre = model.predict(vector_ngram)
    list_text = []
    for i in pre:
      list_text.append(convert_to_text(i))
    if len(text.split()) >=10:
      candidates = [Counter() for i in range(len(list_text) + 9)]
    else :
      candidates = [Counter() for i in range(len(text.split()))]
    for i, sent in enumerate(list_text):
      for j , word in enumerate(sent.split()):
        candidates[i+j].update([word])
    result = ' '.join(i.most_common(1)[0][0] for i in candidates) 
    return result


