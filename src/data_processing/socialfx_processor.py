from datasets import load_dataset
from collections import Counter
import pandas as pd

print("Caricamento dataset in corso...")
dataset = load_dataset("seungheondoh/socialfx-original")
descriptors = [item['text'].lower().strip() for item in dataset['eq']]
counter = Counter(descriptors)
df_top = pd.DataFrame(counter.most_common(20), columns=['Aggettivo', 'Frequenza'])
print(df_top)