# FoodMenu-NER-pt-br

**FoodMenu-NER-pt-br** is a Named Entity Recognition (NER) model tailored for Brazilian Portuguese (pt-br) texts, designed to extract ingredients and their associated quantities from food-related content such as menus and recipes. This model enables precise identification of key entities, facilitating applications like nutritional analysis and health assessments of dishes.

## Key Features

- **Targeted Entity Extraction**: Recognizes two entity classes: "QUANTIDADE" and "INGREDIENTE."
- **Language Optimization**: Specifically engineered for Brazilian Portuguese processing.
- **High Accuracy**: Delivers robust performance in identifying entities with precision.
- **Data Efficiency**: Achieves strong results using a subset of the full dataset, demonstrating effective generalization.

## Project Structure

```plaintext
FoodMenu-NER-pt-br/
├── data/                     # Raw and processed datasets
│   └── dataset.json          # Annotated dataset file
├── model/                    # Trained models and checkpoints
├── scripts/                  # Training and inference scripts
│   ├── evaluate_model.py     # Model evaluation script
│   ├── inference.py          # Inference script for predictions
│   └── train.py              # Model training script
├── .gitignore                # Git ignore file
├── README.md                 # Project documentation
├── requirements.txt          # Project dependencies
```

## Installation

To configure the environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Model Training

Train the model from scratch or fine-tune it further with:

```bash
python scripts/train.py
```

This script executes the training pipeline and saves the optimal model to the `model/` directory.

## Model Evaluation

Assess the model's performance on a validation set using:

```bash
python scripts/evaluate_model.py
```

**Evaluation Metrics:**

![metrics.png](https://i.postimg.cc/BZTCGfcy/metrics.png)

## Performing Inference

Run predictions on custom text inputs with:

```bash
python scripts/inference.py --text "250g de filé de frango grelhado, temperado com limão, azeite e ervas finas. Servido com arroz basmati e legumes assados como abóbora e cenoura"
```

**Inference Example:**

![inference.png](https://i.postimg.cc/RFHx0ZVw/inference.png)

## Dataset Overview

The dataset consists of Brazilian Portuguese texts from food-related sources, annotated with two entity classes:

- **Classes**: `INGREDIENTE`, `QUANTIDADE`
- **Total Annotated Examples**: 243
- **Entity Distribution**:
  - `QUANTIDADE`: 301 instances (21.0%)
  - `INGREDIENTE`: 1130 instances (79.0%)

**Annotated Example:**

```json
{
  "text": "2 porções da Nossa deliciosa galinhada de sobrecoxa desossada, sem pele, sem osso, só o filezinho suculento de coxa, são em média de 300g de galinhada acompanhada de Feijão carioca com pedacinhos de bacon, bem caseirinho e saladinha de alface e tomate. Embalagem Especial Para Os Pratos Executivo. Com Divisória E Selado Para Não Derramar. Nesse combo são duas unidades!",
  "entities": [
    {"start": 0, "end": 9, "label": "QUANTIDADE", "text": "2 porções"},
    {"start": 42, "end": 51, "label": "INGREDIENTE", "text": "sobrecoxa"},
    {"start": 133, "end": 137, "label": "QUANTIDADE", "text": "300g"},
    {"start": 29, "end": 38, "label": "INGREDIENTE", "text": "galinhada"},
    {"start": 166, "end": 180, "label": "INGREDIENTE", "text": "Feijão carioca"},
    {"start": 199, "end": 204, "label": "INGREDIENTE", "text": "bacon"},
    {"start": 236, "end": 242, "label": "INGREDIENTE", "text": "alface"},
    {"start": 245, "end": 251, "label": "INGREDIENTE", "text": "tomate"}
  ]
}
```

## Training Results

- **Base Model**: `pt_core_news_lg`
- **Iterations**: 60
- **Evaluation Metrics**: See metrics image above.

## Author

**Thiago Cunha**
GitHub: [thiagocborghi](https://github.com/thiagocborghi)

If you have any questions or suggestions, feel free to **open an issue** or contribute to the project!

---

## License

This project is released under the MIT License - refer to the [LICENSE](LICENSE) file for details.