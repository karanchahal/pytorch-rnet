mkdir data
cd data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
cd ..
python gen_vocab.py
