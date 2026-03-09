# Tabular-SDG-CTGAN-Evaluation-N46006855

Repository ufficiale dell'attività sperimentale per l'elaborato finale in **Elementi di Intelligenza Artificiale**.
**Università degli Studi di Napoli Federico II** - Corso di Laurea Triennale in Ingegneria Informatica.

## 📌 Descrizione del Progetto
Il progetto analizza la generazione di dati sintetici tabulari utilizzando l'architettura **CTGAN** (Conditional Generative Adversarial Network). L'obiettivo principale è valutare come la variazione del numero di epoche di addestramento influisca sulla fedeltà statistica e sull'utilità dei dati prodotti.

**Dataset utilizzato:** [Bank Marketing Dataset (UCI)](https://archive.ics.uci.edu/dataset/222/bank+marketing)

## 📁 Struttura della Repository
- `src/`: Contiene gli script Python per il training e la valutazione.
- `dataset/`: Include il file originale `bank-additional-full.csv` e i dataset sintetici generati a 50, 100 e 200 epoche.
- `output/`: Galleria completa dei grafici di confronto (Heatmap, Istogrammi, Bar Plot).
- `requirements.txt`: Elenco delle dipendenze necessarie.

## 🛠️ Installazione e Utilizzo
Per riprodurre l'esperimento, clona la repository e installa le librerie necessarie:

```bash
git clone https://github.com/francescov03/Tabular-SDG-CTGAN-Evaluation-N46006855.git table-evaluator
scikit-learn
cd Tabular-SDG-CTGAN-Evaluation-N46006855
pip install -r requirements.txt
