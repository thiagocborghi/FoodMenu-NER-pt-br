import spacy
import json
import sys
import subprocess
from typing import List, Tuple, Dict

def download_model(model_name):
    try:
        spacy.load(model_name)
        print(f"Modelo {model_name} já está instalado.")
    except OSError:
        print(f"Baixando modelo {model_name}...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy-lookups-data"])
        print(f"Modelo {model_name} instalado com sucesso!")

def validate_entities(text: str, entities: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    valid_entities = []
    
    for start, end, label in entities:
        if start < 0 or end > len(text) or start >= end:
            continue
        
        entity_text = text[start:end].strip()
        if not entity_text:
            continue
            
        valid_entities.append((start, end, label))
    
    return valid_entities

def load_data(file_path: str) -> List[Tuple[str, Dict]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    training_data = []
    entity_counts = {}
    
    for annotation in data['annotations']:
        text = annotation['text']
        entities = [(ent['start'], ent['end'], ent['label']) 
                   for ent in annotation['entities']]
        
        valid_entities = validate_entities(text, entities)
        
        if valid_entities:
            training_data.append((text, {'entities': valid_entities}))
            
            for _, _, label in valid_entities:
                entity_counts[label] = entity_counts.get(label, 0) + 1
    
    print("\nRelatório de processamento:")
    print(f"Exemplos totais: {len(data['annotations'])}")
    print(f"Exemplos válidos: {len(training_data)}")
    
    if entity_counts:
        print("\nDistribuição das entidades:")
        total = sum(entity_counts.values())
        for label, count in entity_counts.items():
            print(f"{label}: {count} exemplos ({count/total:.1%})")
    
    return training_data
