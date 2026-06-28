# Smart Vacuum Cleaner — Progetto di Intelligenza Artificiale

**Università degli Studi di Perugia** · Corso di Laurea Triennale in Informatica  
**A.A. 2023/2024** · Materia: Intelligenza Artificiale

**Autori:** Luca Capuccini (347711) · Edoardo Tommasi (342244)

---

## Descrizione

Implementazione del problema del robot aspirapolvere su una matrice N×N di stanze. Il sistema acquisisce la matrice direttamente da un'immagine (digitale o fotografata) tramite una pipeline OCR + rete neurale convoluzionale, poi la risolve con due algoritmi di ricerca: uno non informato e uno informato.

---

## Dominio del Problema

La matrice è composta da stanze con i seguenti simboli:

| Simbolo | Significato | Costo pulizia |
|---------|-------------|---------------|
| `S` | Stanza di partenza | — |
| `F` | Stanza di arrivo | — |
| `C` | Stanza pulita | — |
| `D` | Stanza sporca | 1 |
| `V` | Stanza molto sporca | 3 |
| `X` | Stanza inaccessibile | — |

**Obiettivo:** il robot parte da `S`, pulisce tutte le stanze `D` e `V`, e termina in `F` muovendosi nelle 4 direzioni cardinali senza attraversare stanze `X`.

Un dominio è **invalido** se: manca di un'unica `S` o un'unica `F`, se `F` o una stanza sporca sono irraggiungibili, o se la matrice non è quadrata.

---

## Architettura del Sistema

```
Immagine input
      │
      ▼
┌─────────────────────┐
│  Pipeline OCR/CV    │  OpenCV: rimozione griglia → scala di grigi →
│                     │  binarizzazione → rilevamento contorni → ROI
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Modello CNN        │  TensorFlow/Keras, addestrato su EMNIST
│  (classificazione)  │  riconosce: S, C, X, V, D, F
└────────┬────────────┘
         │  matrice N×N di lettere
         ▼
┌─────────────────────┐
│  Algoritmi di       │  DfsAlt (non informato)
│  Ricerca            │  GreedyAlt (informato, euristica Manhattan)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  GUI Tkinter        │  visualizza immagini di elaborazione,
│                     │  percorso passo-passo, costi, matrici
└─────────────────────┘
```

---

## Modello CNN (OCR)

**File:** `TrainingTestBatch`

Rete convoluzionale a tre strati addestrata sul dataset **EMNIST Letters**, filtrato sui soli 6 caratteri del problema (~900 campioni per classe).

**Architettura:**

```
Conv2D(64, 3×3, ReLU) → MaxPooling(2×2)
Conv2D(128, 3×3, ReLU) → MaxPooling(2×2)
Conv2D(256, 3×3, ReLU) → MaxPooling(2×2)
Flatten → Dense(512, ReLU) → Dropout(0.5)
Dense(25, Softmax)
```

**Training:** ottimizzatore Adam (lr=0.0001), early stopping, data augmentation (rotazioni, shift, zoom).

| Fase | Accuracy | Loss |
|------|----------|------|
| Training | 98.16% | 7.15% |
| Test | 98.02% | 6.87% |

Le accuracy coincidenti indicano assenza di overfitting e underfitting.

**Pipeline di lettura immagine:**
1. Rimozione della griglia con trasformata di Hough probabilistica
2. Scala di grigi, ridimensionamento, blur gaussiano, binarizzazione
3. Rilevamento contorni (`cv2.findContours`)
4. Estrazione ROI per ogni contorno, ridimensionamento a 28×28
5. Predizione con il modello → mapping indice EMNIST → lettera
6. Ordinamento spaziale (per y poi per x) → ricostruzione matrice

**Limiti OCR:** funziona ottimamente su matrici digitali fino a 12×12 (accuratezza 90–100%). Immagini con ombre, griglie storte o molto piccole possono causare errori.

---

## Algoritmi di Ricerca

**File:** `TestMatrix`

### DfsAlt — Ricerca Non Informata

DFS personalizzata con **frontiera ed esplorati locali per ogni nodo child** (non globali). Si sviluppa in profondità estraendo sempre il nodo inserito per primo dalla frontiera locale.

Vengono eseguite due varianti e viene scelta quella con costo percorso minore:

- **Semi-Deterministica:** deterministica, ma passa a mosse casuali quando rileva un loop (countloop ≥ 4)
- **Randomica:** ad ogni passo sceglie con probabilità 60% l'azione deterministica, 40% casuale

Gestione situazioni anomale: quando il robot è bloccato in tutte le direzioni, rimuove nodi dagli esplorati per permettere il backtracking.

**Complessità spaziale:** Θ(n²) rispetto ai nodi esplorati  
**Complessità temporale** (misure empiriche):

| Matrice | Tempo |
|---------|-------|
| 3×3 | ~0.016 s |
| 6×6 | ~0.014 s |
| 12×12 | ~0.359 s |

---

### GreedyAlt — Ricerca Informata

Greedy pesantemente modificata che usa DfsAlt come motore di esplorazione con **euristica dinamica basata sulla distanza di Manhattan**.

**Come funziona l'euristica:**
1. Trova la stanza sporca più vicina alla posizione attuale del robot
2. Costruisce una matrice di valori euristici: per ogni cella, distanza Manhattan verso quella stanza sporca
3. Il robot si sposta verso il vicino con valore euristico minore
4. Dopo aver pulito una stanza, ricalcola la matrice puntando alla prossima sporca più vicina
5. Quando tutte le stanze sono pulite, punta a `F`

Le stanze `X` ricevono euristica ∞. In caso di loop, introduce casualità per uscirne.

**Complessità spaziale:** Θ(n² × n²) (ricalcolo della matrice ad ogni pulizia)  
**Complessità temporale** (misure empiriche):

| Matrice | Tempo |
|---------|-------|
| 3×3 | ~0.001 s |
| 6×6 | ~0.006 s |
| 12×12 | ~0.460 s |

---

## Interfaccia Grafica

GUI Tkinter con tema **Forest Dark**. Funzionalità:
- **Carica Immagine:** apre un'immagine della matrice da file
- **DfsAlt:** esegue la ricerca non informata e mostra i risultati
- **GreedyAlt:** esegue la ricerca informata con euristica e mostra i risultati

L'output include: matrice letta dall'OCR, matrice finale, percorso passo-passo con frontiera/esplorati ad ogni step, costo totale (distanza + pulizia D + pulizia V), tempo OCR e tempo algoritmo.

---

## Dipendenze

```
tensorflow / keras
emnist
numpy
opencv-python (cv2)
Pillow (PIL)
scikit-learn
tkinter (stdlib)
```

Il modello addestrato va salvato come `modello.h5` nella directory del progetto. Il tema grafico richiede `Tema/forest-dark.tcl`.

---

## Limitazioni

- Testato e validato su matrici fino a **12×12**; matrici superiori sono teoricamente supportate ma non verificate
- L'OCR è ottimizzato per immagini digitali; su foto fisiche le performance dipendono dalla qualità dell'illuminazione e della griglia
- DfsAlt e GreedyAlt sono implementazioni custom ("home-made") non equivalenti alle versioni standard in termini di complessità, ma funzionali e con gestione di loop e vicoli ciechi
