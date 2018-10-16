### Demo code

```python
from xy_tf_idf import TfIdfGenerator

input_file = path/to/input_file
output_file = path/to/output_file
idf_dict_path = path/to/idf_dict_path

tfidf = TfIdfGenerator(idf_dict_path)
# generate the idf_dict, if the idf_dict is already there, skip this step
with open(input_file, 'r') as fin:
    tfidf.get_idf_score(fin)

# calculate the tf-idf scores for the given file
tfidf_res = tfidf(input_file)
assert tfidf_res != None
# sort it by tf-idf score in a descending order
tfidf_res.sort(key = lambda x : x[3], reverse = True)
tfidf.write_to_file(tfidf_res, output_file)
```
