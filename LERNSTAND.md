# Lernstand vor Exercise 06

Stand: 20.11.2026. Grundlage: Git-Historie (Commits mit `Co-Authored-By: Claude` = mit Claude
entstanden), der aktuelle Code und die Notebooks.

**Legende:** ✅ von dir · 🤖 von Claude · ⬜ noch offen · 📖 vorgegeben (nur lesen)

---

## Kurzfassung

| Ex | Thema | Von dir | Lücke? |
|----|-------|---------|--------|
| 01 | Notebooks, numpy, Abgabe-Workflow | alles (trivial) | nein |
| 02 | – (nicht im Repo) | – | – |
| 03 | Dataset, DataLoader, Transforms | DataLoader, Transforms, Normalisierung | klein: `__getitem__` |
| 04 | Logistische Regression: Sigmoid, BCE, Backprop, Gradient Descent, Solver | **nichts davon** | **groß**: 6 TODOs noch offen |
| 05 | Neuronales Netz: Layer, Affine, ReLU, Momentum | **nichts davon** | **groß** |
| 06 | Hyperparameter-Tuning, Regularisierung, ≥ 48 % auf CIFAR-10 | – | – |

Exercise 04 und 05 bilden zusammen den Kern der Vorlesung: **Forward Pass → Loss → Backward
Pass (Kettenregel) → Update-Schritt**. Exercise 06 baut darauf auf, ohne es neu zu erklären.

---

## Exercise 01 – Einführung

**Was du gemacht hast:** ✅ `Dummy.forward` (`x = 100`), Notebook durchgearbeitet, Abgabe.

**Lernziel:**
- Jupyter-Notebooks, Imports, `%autoreload`
- numpy-Grundlagen: Arrays, Shapes, Broadcasting, Vektorisierung statt Schleifen
- Ablauf: Code in `exercise_code/` → Modell speichern → `submit_exercise` → Zip hochladen

**In Ex06:** Derselbe Ablauf am Ende (`save_pickle`, `submit_exercise`). numpy-Broadcasting
steckt überall drin, z. B. `x @ W + b` addiert `b` auf jede Zeile.

---

## Exercise 03 – Daten: Dataset, DataLoader, Transforms

**Was du gemacht hast:**
- 🤖 `ImageFolderDataset.__len__` und `__getitem__`, Notizen zu `make_dataset`
  (Commit `d342918`). Den `__getitem__` hast du danach selbst vereinfacht.
- ✅ `DataLoader.__iter__`: Shuffle über Indizes, Batches bauen, `drop_last`,
  Liste von Dicts → Dict von Arrays, `yield` (Commit `e28016b`)
- ✅ `DataLoader.__len__`: `//` plus Restbatch je nach `drop_last`
- ✅ `RescaleTransform`: lineares Umskalieren von [0, 255] nach [0, 1]
- ✅ `compute_image_mean_and_std`: Mittelwert und Std pro Kanal, `axis=(0, 1, 2)`
- ✅ `NormalizeTransform`: `(x - mean) / std`

> Selbstcheck: Die Kommentare im DataLoader sind sehr ausführlich. Wenn du ihn ohne Vorlage
> nicht nochmal schreiben könntest, zählt er eher halb.

**Lernziel:**
- Trennung von *Dataset* (ein Sample per Index) und *DataLoader* (Batches, Shuffle)
- Lazy Loading von der Festplatte vs. alles im RAM (`MemoryImageFolderDataset`)
- Vorverarbeitung: Rescale, Normalisierung auf Mittelwert 0 und Std 1, Transforms als
  aufrufbare Klassen
- Aufteilung in Train, Val und Test (`select_split`)

**In Ex06:**
- Dataset und DataLoader sind fertig vorgegeben und entsprechen deinem Code aus Ex03.
  Rescale, Normalize und `compute_image_mean_and_std` stehen jetzt in
  `exercise_06/exercise_code/data/image_folder_dataset.py`.
- **Neu:** `FlattenTransform` macht aus einem Bild (32, 32, 3) einen Vektor (3072,), weil
  ein Fully-Connected-Netz Vektoren braucht. Das ist dasselbe `reshape(N, -1)` wie im
  Affine-Layer.
- **Neu:** `ComposeTransform` (mehrere Transforms hintereinander) und `RandomHorizontalFlip`
  (Data Augmentation). Welche Augmentations du nutzt, ist selbst ein Hyperparameter.
- Train, Val und Test werden jetzt richtig genutzt: auf Val wird getunt, Test nur **einmal**
  am Ende.

---

## Exercise 04 – Logistische Regression (ein Neuron)

**Was du gemacht hast:**
- 🤖 `BCE.forward` (Commit `ade5502`)
- 🤖 Notizen (Erklärung, Pseudocode, Stolperfallen) zu allen übrigen TODOs
- ⬜ `Classifier.sigmoid`
- ⬜ `Classifier.forward` (inkl. Cache)
- ⬜ `Classifier.backward` (dL/dW)
- ⬜ `BCE.backward`
- ⬜ `Optimizer.step` (Gradient Descent)
- ⬜ `Solver._step`

Im Repo sind diese 6 TODOs noch leer. Falls du anderswo abgegeben hast, stimmt das nicht mit
dem Repo überein.

**Lernziel**, also das Fundament für alles Weitere:
- **Modell:** `ŷ = σ(XW)`, die Sigmoid macht aus einem Score eine Wahrscheinlichkeit
- **Loss:** Binary Cross-Entropy `-[y log ŷ + (1-y) log(1-ŷ)]`, gemittelt über N
- **Backprop:** Kettenregel `dL/dW = dL/dŷ · dŷ/ds · ds/dW`. Hübsches Ergebnis:
  Sigmoid und BCE zusammen ergeben `(ŷ - y)/N`.
- **Cache:** Im Forward speichern, was der Backward braucht
- **Gradient Descent:** `W ← W - lr · dW`
- **Solver:** die Trainingsschleife pro Batch: forward → loss → backward → step

**In Ex06:**
- Der `Solver` in Ex06 ist die erweiterte Version deines Ex04-Solvers: `_step` macht genau
  forward, loss, backward und step. Neu sind Validierung, Early Stopping und `lr_decay`.
- Die Klasse `SGD` in `networks/optimizer.py` ist genau dein offenes `Optimizer.step`.
- Die Kettenregel aus `Classifier.backward` wird in `ClassificationNet.backward` nur über
  mehrere Layer wiederholt.
- `BCE` steht noch in `loss.py`, für CIFAR-10 (10 Klassen) wird aber Cross-Entropy genutzt.

---

## Exercise 05 – Neuronale Netze aus Bausteinen

**Was du gemacht hast:**
- 🤖 `Sigmoid.forward` und `backward`
- 🤖 `Relu.forward` und `backward`
- 🤖 `affine_forward` und `affine_backward`
- 🤖 `SGDMomentum._update`
- ✅ Notebook ausgeführt
- ⬜ Reasoning-Frage „Warum ist der erste Loss ≈ −log(0.1) ≈ 2.30?“ ist noch nicht im
  Notebook beantwortet
- 📖 `CrossEntropyFromLogits` und `Adam` waren vorgegeben, nur lesen

**Lernziel:**
- **Modularer Backprop:** Jeder Layer hat `forward(x) → (out, cache)` und
  `backward(dout, cache) → dx`. Das Netz verkettet nur Layer, und die Kettenregel ergibt
  sich von selbst.
- **Aktivierungen** sind element-weise, deshalb ist der Backward `dout * f'(x)` mit `*` und
  nicht mit `@`.
  - Sigmoid: `σ' = σ(1−σ)` ist höchstens 0.25 → Vanishing Gradients
  - ReLU: `dout * (x > 0)`, der Gradient wird durchgelassen oder blockiert
- **Affine-Layer** `z = XW + b`:
  - `dx = dout @ Wᵀ`, `dW = Xᵀ @ dout`, `db = Σ_N dout`
  - Trick zum Herleiten: Jeder Gradient hat dieselbe Shape wie seine Variable.
  - Kein `/N` hier, das steckt schon im Loss.
- **Cross-Entropy mit Softmax:** Der Gradient ist `(p − onehot(y)) / N`.
- **Optimizer:**
  - SGD
  - Momentum: `v = βv − lr·dw`, `w += v`
  - Adam: Momentum plus adaptive Schrittweite pro Parameter plus Bias-Korrektur
- **Sanity Check:** Der erste Loss liegt bei ≈ ln(10) ≈ 2.30, weil das Netz bei
  zufälliger Initialisierung jeder Klasse etwa 1/10 gibt. **Prüfungsrelevant!**

**In Ex06:** Hier fließt fast alles ein.
- `Sigmoid`, `Relu`, `affine_*` sind fertig vorgegeben. Die Aufgabe im Notebook (Zelle
  1.3): *„Verstehe, warum element-weises Produkt statt Matrixprodukt“*. Das ist genau der
  Ex05-Stoff, den Claude geschrieben hat.
- ⬜ **Neue TODOs in `networks/layer.py`: `LeakyRelu` und `Tanh`, jeweils Forward und
  Backward.** Sie funktionieren wie ReLU bzw. Sigmoid aus Ex05:
  - LeakyReLU: `x` für x > 0, sonst `slope · x`. Die Ableitung ist 1 bzw. `slope`.
  - Tanh: Cache den Output. Die Ableitung ist `1 − tanh²`.
- `ClassificationNet` ist das N-Layer-Netz aus Ex05: eine Schleife über affine und
  Aktivierung, im Backward rückwärts. Die Aktivierung ist austauschbar, das ist ein
  Hyperparameter.
- `CrossEntropyFromLogits`, `SGD`, `sgd_momentum`, `Adam` sind vorgegeben. Sie werden in
  den Zellen 1.4 und 1.5 nur angeschaut und später beim Tuning als Optimizer gewählt.
- Debugging-Schritt 2.1 „erst ein Sample overfitten“: Der Loss muss bei ≈ 2.3 starten und
  gegen 0 gehen.

---

## Exercise 06 – Was ist wirklich neu?

Das Bestehen hängt diesmal nicht an Unit-Tests, sondern an **≥ 48 % Test-Accuracy** auf dem
Server.

| Neues Konzept | Wo im Code |
|---|---|
| L2-Regularisierung (Weight Decay): `L* = L + λ Σ W²`, Gradient `+ 2λW` | `ClassificationNet.forward` (`self.reg`) und `backward` (`+ 2 * self.reg_strength * W`) |
| Hyperparameter: lr, reg, hidden_size, num_layer, Aktivierung, Optimizer, Batch-Size, Epochen | Notebook Kapitel 2 |
| Grid Search vs. Random Search (lr und reg **logarithmisch** sampeln) | `hyperparameter_tuning.py` |
| Early Stopping mit `patience`; die besten Val-Parameter werden zurückgeschrieben | `Solver.update_best_loss`, `Solver.train` |
| Learning-Rate-Decay pro Epoche | `Solver.train` (`self.opt.lr *= self.lr_decay`) |
| Data Augmentation | `RandomHorizontalFlip`, `ComposeTransform` |
| Overfitting erkennen: Train-Loss sinkt, Val-Loss steigt | Loss-Plots im Notebook |
| Test-Set nur einmal benutzen | Notebook Kapitel 4 |
| Optional: eigenes Netz `MyOwnNetwork` | `classification_net.py` |

---

## Empfohlener Plan, um die Lücke zu schließen

In dieser Reihenfolge, jeweils ohne Claude, nur mit den Notizen in den TODO-Blöcken:

1. **Ex04 fertig machen:** die 6 offenen TODOs. Es ist wenig Code, aber genau das Fundament
   (Kettenregel bei einem einzelnen Neuron). Die Notizen stehen schon im Code. Die
   Notebook-Tests prüfen dich.
2. **Ex05 nachvollziehen:** `affine_backward` auf Papier mit Shapes herleiten
   ((N, M) @ (M, D) usw.) und die Reasoning-Frage zu 2.30 ins Notebook schreiben.
3. **Ex06:** `LeakyRelu` und `Tanh` selbst schreiben. Das ist die direkte Probe, ob Schritt
   2 sitzt.
4. **Ex06:** `ClassificationNet.forward` und `backward` Zeile für Zeile lesen, dann das
   Hyperparameter-Tuning.

## Selbsttest (ohne nachzuschauen beantworten)

1. Warum `dout * s * (1 − s)` und nicht `dout @ …` im Sigmoid-Backward?
2. Welche Shape hat `dW` im Affine-Layer, und wie bekommst du sie aus `x` und `dout`?
3. Warum wird `db` über `axis=0` summiert?
4. Wo kommt das `1/N` im Gradienten her, und warum gehört es **nicht** in `affine_backward`?
5. Warum ist der erste Loss ≈ 2.30?
6. Was macht Momentum anders als SGD, und was passiert im allerersten Schritt?
7. Warum sampelt man die Learning Rate logarithmisch und nicht linear?
8. Woran erkennst du Overfitting, und welche zwei Mittel aus Ex06 helfen dagegen?
9. Warum darf man auf dem Test-Set nicht tunen?
