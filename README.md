Order to Follow: 

Data Collection: Data Scraper and PDF Merger. These downloaded all the corporate sustainability reports and standards, and combined standards to one larger document. 

Data Cleaning: ESG Text Cleaner. This normalized the text specific for accounting applications, meaning select special characters were not removed vs traditional cleaning.

Models: 

Baseline Models: These were the first pass models with very limited performance combined to one file. 

msci_bert_messy_model: This was the first attempt at a refined model, using BERT and unprocessed data. It was very computationally inefficient and a new baseline. 

msci_bert_model: This was the eventual preferred model, using cleaned text and BERT

msci_finbert_model: Although FINBERT is specific for financial text, sustainability reporting has a lot of non-financial aspects, which could be why it underperformed traditional BERT. 

Model Result Sampling (only part of Model 2 shown here due to wifi constraints)

folder_v1 for model 1
folder_v2 for model 2
folder_v3 for model 3
