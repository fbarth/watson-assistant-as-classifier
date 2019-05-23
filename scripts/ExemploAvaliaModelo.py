import pandas as pd
import numpy as np
import json
import sys
import csv
import ibm_watson
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

assistant = ibm_watson.AssistantV1(
    version='2019-02-28',
    iam_apikey='{api_key}',
    url='https://gateway.watsonplatform.net/assistant/api')

result = pd.DataFrame(columns=['class', 'predicted', 'confidence'])
with open('../data/test.csv', newline='') as f:
  reader = csv.reader(f)
  count=0
  for row in reader:
    response = assistant.message(
        workspace_id='233eb6a2-4977-45a3-93db-f79a7c21150a',
        input={'text': row[0]}).get_result()
    try:
        result.loc[-1] = [row[1], response['intents'][0]['intent'], response['intents'][0]['confidence']]
        count=count+1
        print(count, [row[1], response['intents'][0]['intent'], response['intents'][0]['confidence']])
        result.index = result.index + 1
        result = result.sort_index()
    except:
        None

print(sum(result['class'] == result['predicted']) / len(result))
m = confusion_matrix(result['class'], result['predicted'])
print(m)
accr = accuracy_score(result['class'], result['predicted'])
print(accr)