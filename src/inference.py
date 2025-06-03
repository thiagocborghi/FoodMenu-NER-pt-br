import os
import spacy
from spacy import displacy
import logging
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from typing import List, Dict
import argparse
import webbrowser

logging.basicConfig(
  level=logging.INFO,
  format="%(message)s", 
  handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("food_ner")
console = Console()

class FoodNERPredictor:
  def __init__(self, model_path: str):
      path = Path(model_path)
      if not path.exists():
          raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
          
      logger.info(f"Carregando modelo de: {model_path}")
      self.nlp = spacy.load(model_path)
      self.nlp.select_pipes(enable=["ner"])

  def predict_text(self, text: str) -> List[Dict]:
      doc = self.nlp(text)
      entities = []
      
      for ent in doc.ents:
          entity = {
              "text": ent.text,
              "label": ent.label_,
              "start": ent.start_char,
              "end": ent.end_char
          }
          entities.append(entity)
      
      return entities

  def format_prediction(self, text: str, entities: List[Dict]) -> None:
      if not entities:
          console.print(Panel("Nenhuma entidade encontrada", style="red"))
          return

      console.log("\n\n")
      console.print(Panel(text, title="Texto Analisado", style="cyan"))

      table = Table()
      table.add_column("Tipo", style="cyan", justify="center")
      table.add_column("Entidade", style="not dim")
      
      entities = sorted(entities, key=lambda x: (x["label"], x["start"]))
      
      for entity in entities:
          entity_color = "yellow" if entity["label"] == "QUANTIDADE" else "green"
          label_color = "blue3" if entity["label"] == "QUANTIDADE" else "cyan"
          table.add_row(Text(entity["label"], style=label_color), Text(entity["text"], style=entity_color))

      console.print(table)
      console.log("\n\n")


  def display_prediction(self, text: str):
        doc = self.nlp(text)
        return displacy.render(doc, style="ent")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--model', type=str, default="model/food_item_ner_pt")
    parser.add_argument('--display', type=str, default="cli")
    args = parser.parse_args()
    
    try:
        predictor = FoodNERPredictor(args.model)
        entities = predictor.predict_text(args.text)
        
        if args.display == "cli":
            predictor.format_prediction(args.text, entities)
        else:
            html_output = predictor.display_prediction(args.text)
            temp_html = 'temp_prediction.html'
            with open(temp_html, 'w', encoding='utf-8') as f:
                f.write(html_output)
            webbrowser.open('file://' + os.path.realpath(temp_html))

    except Exception as e:
        logger.error(f"Erro durante inferência: {str(e)}")
        raise

if __name__ == "__main__":
  main()