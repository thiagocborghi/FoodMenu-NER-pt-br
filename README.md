# FoodMenu-NER-pt-br

**FoodMenu-NER-pt-br** is a Named Entity Recognition (NER) model specialized for Brazilian Portuguese (pt-br) texts, designed to extract ingredients and their quantities from food-related content such as menus and recipes.

## Technical Specifications

### Model Architecture
- **Base Model**: `pt_core_news_lg` (spaCy)
- **Pipeline**: Preserves all components except "ner", "trf_wordpiecer" and "trf_tok2vec"
- **Fine-tuning**: Transfer Learning focused on NER layer

### Hyperparameters
- **Number of Iterations**: 60
- **Batch Size**: Dynamic (4 to 16, using compounding)
- **Dropout Rate**: 0.3

### Entity Classes
- `QUANTIDADE`: Identification of measurements and quantities
- `INGREDIENTE`: Recognition of food ingredients

## Project Structure

```plaintext
FoodMenu-NER-pt-br/
├── data/
│   └── dataset.json         
├── model/
│   └── food_item_ner_pt/   
├── src/
│   ├── utils.py             
│   ├── evaluate_model.py    
│   ├── inference.py         
│   └── train.py             
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

### Statistics
- **Total Examples**: 243
- **Data Split**: 80% training, 20% validation
- **Entity Distribution**:
  - `QUANTIDADE`: 301 instances
  - `INGREDIENTE`: 1130 instances

### Data Format
```json
{
  "text": "500g de macarrão com molho",
  "entities": [
    {
      "start": 0,
      "end": 4,
      "label": "QUANTIDADE",
      "text": "500g"
    },
    {
      "start": 8,
      "end": 16,
      "label": "INGREDIENTE",
      "text": "macarrão"
    }
  ]
}
```

## Training and Evaluation

To train the model, run:
```bash
python src/train.py
```

To evaluate the trained model:
```bash
python src/evaluate_model.py
```

### Evaluation Metrics

**Performance Metrics:**

![metrics.png](https://i.postimg.cc/BZTCGfcy/metrics.png)

## Inference

```python
import spacy

nlp = spacy.load("model/food_item_ner_pt")

text = "250g de filé de frango grelhado, temperado com limão, azeite e ervas finas. Servido com arroz basmati e legumes assados como abóbora e cenoura"
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.label_}: {ent.text}")
```

You can also run predictions using the command line:

```bash
python src/inference.py --text "250g de filé de frango grelhado, temperado com limão, azeite e ervas finas. Servido com arroz basmati e legumes assados como abóbora e cenoura"
```

**Inference Example:**

![inference.png](https://i.postimg.cc/RFHx0ZVw/inference.png)

## Known Limitations
- Model optimized only for culinary texts
- May struggle with regional culinary slang
- Performance not tested on very long texts

## License

This project is released under the MIT License - refer to the [LICENSE](LICENSE) file for details.