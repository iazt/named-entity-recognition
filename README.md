# Named Entity Recognition


Sequence Labelling is a task of Natural Language Processing (NLP). Its main objective is to tag a sequence of tokens contained in a sentence.

On the other hand, Named Entity Recognition (NER) is a subtask of Sequence Labelling and its principal goal is to identify and classify named entity mentions in unstructured text into pre-defined categories such as person names, locations, time expressions, organizations, etc.

In this repository, different solutions are compared to solve a NER problem using a spanish database.

This is a task for the NLP course, CC6205 of the University of Chile. [Here](https://github.com/iazt/named-entity-recognition/blob/master/baseline.ipynb), you can find a baseline for the task, which is a basic solution created by the assistant professor Pablo Badilla. 

Data: [CoNLL 2002 Spanish](https://www.clips.uantwerpen.be/conll2002/ner/).



## Results in Test set

<table class="tg">
  <tr>
    <th class="tg-baqh" rowspan="2"><br>Model / Metric</th>
    <th class="tg-baqh" colspan="3">Macro AVG</th>
  </tr>
  <tr>
    <td class="tg-c3ow">F1 </td>
    <td class="tg-c3ow">Precision</td>
    <td class="tg-c3ow">Recall</td>
  </tr>
  <tr>
    <td class="tg-c3ow">BILMST (hidden dim = 512, layers = 3, dropout = 0.2)</td>
    <td class="tg-c3ow">0.6664 +- 0.026</td>
    <td class="tg-c3ow">0.745 +-  0.0115</td>
    <td class="tg-c3ow">0.615 +- 0.035</td>
  </tr>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
</table>
