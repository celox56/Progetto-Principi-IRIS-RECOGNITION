# Iris Recognition con IrisCode

Progetto universitario di riconoscimento dell’iride basato su una versione semplificata dell’algoritmo di Daugman, sviluppato in Python.  
L’obiettivo è:

- estrarre l’**IrisCode** da immagini dell’occhio,
- calcolare le **distanze di Hamming** tra codici,
- analizzare le prestazioni del sistema tramite **istogrammi**, **ROC curve** e **metriche quantitative**.

---

## 📚 Contenuti

- [Descrizione del progetto](#descrizione-del-progetto)
- [Algoritmo implementato](#algoritmo-implementato)
  - [1. Segmentazione dell’iride](#1-segmentazione-delliride)
  - [2. Normalizzazione (Rubber Sheet Model)](#2-normalizzazione-rubber-sheet-model)
  - [3. Estrazione dellIrisCode](#3-estrazione-delliriscode)
  - [4. Distanza di Hamming e decisione](#4-distanza-di-hamming-e-decisione)
- [Struttura del dataset](#struttura-del-dataset)
- [Struttura del progetto](#struttura-del-progetto)
- [Requisiti](#requisiti)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Output generati](#output-generati)
- [Scelte di visualizzazione dei risultati](#scelte-di-visualizzazione-dei-risultati)
- [Limiti e possibili estensioni](#limiti-e-possibili-estensioni)
- [Riferimenti e crediti](#riferimenti-e-crediti)

---

## 🧠 Descrizione del progetto

L’iride è uno dei tratti biometrici più affidabili:

- è **unica** per ogni individuo (anche nei gemelli monozigoti),
- è **stabile nel tempo**,
- è **protetta** da cornea e palpebre,
- è **difficile da falsificare**.

In questo progetto:

- si lavora su un **dataset di immagini oculari** organizzato per persona,
- si implementa in Python una pipeline completa di **iris recognition**:
  - segmentazione dell’iride,
  - normalizzazione tramite *rubber sheet model*,
  - estrazione di un **IrisCode binario** e relativa maschera,
  - confronto tramite **distanza di Hamming**,
- si analizzano le prestazioni tramite:
  - distanze **genuine** (stessa persona) e **impostor** (persone diverse),
  - **istogrammi bilanciati**,
  - **curva ROC** e **AUC**,
  - un semplice **K-Means** sulle distanze come supporto esplorativo.

---

## ⚙️ Algoritmo implementato

Tutto il flusso è implementato nello script Python (ad es. `iris_recognition.py`).

### 1. Segmentazione dell’iride

Funzione principale: `segment_iris(img)`

- converte l’immagine in **scala di grigi** e applica un **median blur**;
- usa `cv2.HoughCircles` per rilevare il **cerchio della pupilla**, assumendo:
  - pupilla = regione più scura,
  - parametri scelti per min/max raggio;
- stima il **cerchio dell’iride** come un cerchio più grande, centrato sulla stessa posizione;
- in caso di fallimento di Hough:
  - usa un metodo di emergenza (centro dell’immagine + raggio approssimato);
- output:
  - un cerchio per la **pupilla** `(xp, yp, rp)`
  - un cerchio per il **bordo dell’iride** `(xi, yi, ri)`.

L’anello tra questi due cerchi è la regione ricca di informazione biometrica.

---

### 2. Normalizzazione – Rubber Sheet Model

Funzione: `normalize_iris(img, pupil, iris, radial_res, angular_res)`

Implementa il **rubber sheet model** di Daugman:

- l’anello dell’iride viene “srotolato” in coordinate polari;
- per ogni angolo `θ` e per ogni raggio normalizzato `r ∈ [0, 1]`:
  - calcola il punto corrispondente tra bordo pupilla e bordo iride;
  - campiona l’intensità tramite `ndimage.map_coordinates`;
- il risultato è una matrice `radial_res × angular_res` (default: `64 × 256`).

Questa rappresentazione:

- rende le iridi **confrontabili** anche se la pupilla è più o meno dilatata,
- porta tutte le immagini nella **stessa dimensione standard**.

---

### 3. Estrazione dell’IrisCode

Funzione: `gabor_filtering(polar, sigma=2.5)`

Step concettuale:

1. Applica un filtro di tipo **Gabor-like** usando:
   - `gaussian_filter` → parte “reale”,
   - `gaussian_laplace` → parte “immaginaria” (per la maschera).
2. Soglia la parte reale:
   - se valore > media → bit = 1
   - se valore ≤ media → bit = 0
3. Costruisce:
   - `iris_code`: vettore binario (flatten della matrice),
   - `mask`: vettore binario che indica quali bit sono **affidabili** (niente riflessi, ciglia, ecc.).

La maschera viene poi usata per considerare solo i bit “validi” nel confronto.

---

### 4. Distanza di Hamming e decisione

Funzioni principali:

- `hamming_distance(code1, mask1, code2, mask2)`
- `compute_distances(codes_dict)`
- `compute_roc(genuine, impostor)`
- `compute_auc(fpr, tpr)`

Pipeline:

1. **Calcolo IrisCode per tutte le immagini**  
   Funzione: `compute_all_codes(dataset_path)`  
   Restituisce un dizionario:  
   `persona → lista di (path_immagine, iris_code, mask)`.

2. **Distanze genuine**  
   Per ogni persona:
   - fa tutte le combinazioni a coppie tra le sue immagini,
   - calcola la distanza di Hamming usando solo i bit validi in entrambe le maschere.

3. **Distanze impostor**  
   - prende tutte le coppie tra persone diverse,
   - calcola distanze di Hamming,
   - opzionalmente sottocampiona fino a `MAX_IMPOSTOR_PAIRS` (default: 5000).

4. **Valori attesi**:
   - genuine: distanza tipicamente tra ~0.10 e ~0.35,
   - impostor: distanza attorno a ~0.5 (comportamento casuale).

5. **ROC curve**  
   Varia la soglia `t` da 0 a 1 e per ogni valore calcola:
   - TPR (True Positive Rate) = genuine ≤ t
   - FPR (False Positive Rate) = impostor ≤ t  
   Da qui:
   - si disegna la **curva ROC**,
   - si calcola l’**AUC** con integrazione numerica.

---

## 📁 Struttura del dataset

Il codice si aspetta un dataset organizzato così:

```text
Eye database/
├── 001/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── 002/
│   ├── img1.jpg
│   └── ...
├── personA/
│   ├── ...
└── ...matplotlib


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

