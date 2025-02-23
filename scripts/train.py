import spacy
import random
from pathlib import Path
from spacy.training import Example
from spacy.util import minibatch, compounding
from tqdm import tqdm
from utils import load_data, download_model

MODEL_NAME = "pt_core_news_lg"

def train_model(training_data, output_dir, n_iter=60):
    download_model(MODEL_NAME)
    
    print(f"\nCarregando modelo base {MODEL_NAME}...")
    nlp = spacy.load(MODEL_NAME)
    
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe('ner')
    
    for _, annotations in training_data:
        for ent in annotations.get('entities'):
            label = ent[2]
            if label not in ner.labels:
                ner.add_label(label)
    
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    examples = []
    for text, annots in training_data:
        try:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annots)
            examples.append(example)
        except Exception as e:
            print(f"Erro ao processar exemplo: {text[:50]}... - {str(e)}")
    
    dropout_rate = 0.3  
    batch_size_start = 4
    batch_size_end = 16
    
    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = nlp.begin_training()
        
        print("\nIniciando treinamento...")
        for itn in tqdm(range(n_iter)):
            random.shuffle(examples)
            losses = {}

            batches = minibatch(examples, size=compounding(batch_size_start, batch_size_end, 1.001))
            for batch in batches:
                nlp.update(batch, drop=dropout_rate, losses=losses, sgd=optimizer)
            
            if (itn + 1) % 5 == 0:
                print(f"Iteration {itn+1}, Losses: {losses}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"\nModelo salvo em {output_dir}")

    return nlp

if __name__ == "__main__":
    DATA_PATH = "data/dataset.json"
    OUTPUT_DIR = "model/food_item_ner_pt"
    
    print("Carregando dados de treinamento...")
    training_data = load_data(DATA_PATH)
    
    if not training_data:
        print("Erro: Nenhum dado válido para treinamento!")
    else:
        nlp = train_model(training_data, OUTPUT_DIR, n_iter=60)
