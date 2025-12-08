Iris Recognition Project — README
📌 Descrizione del progetto
Questo progetto ha l’obiettivo di analizzare un dataset di immagini dell’iride e applicare tecniche di riconoscimento biometrico basate sulla Hamming Distance.
L’obiettivo principale è verificare la capacità dell’algoritmo di distinguere tra iridi della stessa persona e di persone diverse, valutando l’efficacia tramite grafici e misure statistiche.

📁 Struttura del progetto
Il repository contiene i file principali suddivisi in due blocchi:
1. Processing & Feature Extraction
File che si occupano di:
Caricare le immagini del dataset


Segmentare pupilla e iride


Normalizzare l’immagine


Estrarre l’IrisCode


Calcolare la distanza di Hamming tra coppie di immagini


2. Analisi e Visualizzazioni
File e notebook dedicati alla creazione dei grafici:
Distribuzione delle distanze genuine vs impostor


Confronto tra le distanze medie


ROC curve


Eventuale clustering opzionale


Questi grafici servono a valutare se l’algoritmo distingue correttamente le persone.

🧠 Tecnologia utilizzata
Python


Librerie principali:


OpenCV (image processing)


NumPy


SciPy


scikit-learn


matplotlib


pandas



📊 Dataset
Il dataset utilizzato è composto da oltre 200 immagini di iridi reali, appartenenti sia alla stessa persona che a persone diverse.
La grande variabilità delle iridi è ideale per studiare:
Similarità intra-persona


Differenze inter-persona


Andamento delle distanze biometriche



🔍 Obiettivi raggiunti
Nel progetto abbiamo:
Processato tutte le immagini del dataset


Generato gli IrisCode


Confrontato tutte le coppie possibili


Calcolato le distanze di Hamming


Prodotto grafici chiari e leggibili per valutare il sistema


Dimostrato che l’algoritmo distingue correttamente iridi genuine e impostor



🗂️ File principali
processing.py — Estrazione delle feature e IrisCode


analysis.py — Calcolo distanze e grafici


dataset/ — Immagini dell’iride


results/ — Grafici finali e file CSV



📌 Come eseguire il codice
Installare le dipendenze:


pip install -r requirements.txt

Assicurarsi che la cartella dataset/ contenga le immagini.


Eseguire:


python processing.py
python analysis.py


📚 Autori
Progetto realizzato nell’ambito del corso Principi e Modelli della Percezione.
Team: [Inserisci i nomi del gruppo].

