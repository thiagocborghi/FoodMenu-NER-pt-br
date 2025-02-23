import spacy
import random
from utils import load_data
from tabulate import tabulate
from rich.console import Console
from rich.table import Table

console = Console()

def evaluate_model(nlp, test_data):
    metrics_counter = {
        "QUANTIDADE": {"tp": 0, "fp": 0, "fn": 0},
        "INGREDIENTE": {"tp": 0, "fp": 0, "fn": 0}
    }
    
    debug_info = {
        "QUANTIDADE": {
            "total_predicted": 0,
            "total_gold": 0,
            "matched": []
        },
        "INGREDIENTE": {
            "total_predicted": 0,
            "total_gold": 0,
            "matched": []
        }
    }
    
    console.print("\n[bold cyan]Avaliando exemplos...[/bold cyan]")
    
    for text, annotations in test_data:
        try:
            pred_doc = nlp(text)
            
            pred_entities = {}
            gold_entities = {}
            
            for ent in pred_doc.ents:
                key = (ent.start_char, ent.end_char)
                pred_entities[key] = ent.label_
                debug_info[ent.label_]["total_predicted"] += 1
            
            for start, end, label in annotations.get('entities', []):
                key = (start, end)
                gold_entities[key] = label
                debug_info[label]["total_gold"] += 1
            
            for ent_type in metrics_counter.keys():
                for span, label in pred_entities.items():
                    if label == ent_type:
                        if span in gold_entities and gold_entities[span] == ent_type:
                            metrics_counter[ent_type]["tp"] += 1
                            debug_info[ent_type]["matched"].append((text[span[0]:span[1]], span))
                        else:
                            metrics_counter[ent_type]["fp"] += 1
                
                for span, label in gold_entities.items():
                    if label == ent_type and (span not in pred_entities or pred_entities[span] != ent_type):
                        metrics_counter[ent_type]["fn"] += 1
            
        except Exception as e:
            console.print(f"[red]Erro ao avaliar exemplo:[/red] {str(e)}")
    
    console.print("\n[bold yellow]Debug Info[/bold yellow]")
    for ent_type, info in debug_info.items():
        console.print(f"\n[bold]{ent_type}[/bold]")
        console.print(f"Total previsto: {info['total_predicted']}")
        console.print(f"Total gold: {info['total_gold']}")
        console.print(f"Matches encontrados: {len(info['matched'])}")
    
    table = Table(title="Métricas por Tipo de Entidade", title_style="bold yellow")
    table.add_column("Entidade", style="bold cyan")
    table.add_column("Precisão", style="bold green")
    table.add_column("Recall", style="bold magenta")
    table.add_column("F1-Score", style="bold blue")
    
    for ent_type, counts in metrics_counter.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        
        console.print(f"\n[bold]{ent_type} - Contadores:[/bold]")
        console.print(f"True Positives (TP): {tp}")
        console.print(f"False Positives (FP): {fp}")
        console.print(f"False Negatives (FN): {fn}")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        table.add_row(
            ent_type,
            f"{precision:.2%}",
            f"{recall:.2%}",
            f"{f1:.2%}"
        )
    
    # console.print(table)
    



    # Calcular métricas gerais
    # Calcular métricas gerais
    # Calcular métricas gerais
    total_tp = sum(counts["tp"] for counts in metrics_counter.values())
    total_fp = sum(counts["fp"] for counts in metrics_counter.values())
    total_fn = sum(counts["fn"] for counts in metrics_counter.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    general_metrics = [
        ["Precisão", f"{overall_precision:.2%}"],
        ["Recall", f"{overall_recall:.2%}"],
        ["F1-Score", f"{overall_f1:.2%}"]
    ]
    
    # console.print("\n[bold yellow]Métricas Gerais[/bold yellow]")
    console.print("\n\n")
    console.print(tabulate(general_metrics, headers=["Métrica", "Valor"], tablefmt="fancy_grid"))
    console.print("\n\n")

if __name__ == "__main__":
    MODEL_PATH = "../model/food_item_ner"
    DATA_PATH = "../data/dataset.json"
    
    console.print("\n[bold cyan]Carregando modelo treinado...[/bold cyan]")
    nlp = spacy.load(MODEL_PATH)
    
    console.print("\n[bold cyan]Carregando dados de teste...[/bold cyan]")
    test_data = load_data(DATA_PATH)[-49:]  

    evaluate_model(nlp, test_data)
